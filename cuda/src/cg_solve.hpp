#ifndef _cg_solve_hpp_
#define _cg_solve_hpp_

//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#include <cmath>
#include <limits>

#include <Vector_functions.hpp>
#include <mytimer.hpp>

#include <outstream.hpp>

namespace miniFE {

template<typename Scalar>
void print_vec(const std::vector<Scalar>& vec, const std::string& name)
{
  for(size_t i=0; i<vec.size(); ++i) {
    std::cout << name << "["<<i<<"]: " << vec[i] << std::endl;
  }
}
template<typename Scalar>
void print_cuda_vec(const thrust::device_vector<Scalar>& vec, const std::string& name)
{
  for(size_t i=0; i<vec.size(); ++i) {
    std::cout << name << "["<<i<<"]: " << vec[i] << std::endl;
  }
}

template<typename VectorType>
bool breakdown(typename VectorType::ScalarType inner,
               const VectorType& v,
               const VectorType& w)
{
  typedef typename VectorType::ScalarType Scalar;
  typedef typename TypeTraits<Scalar>::magnitude_type magnitude;

//This is code that was copied from Aztec, and originally written
//by my hero, Ray Tuminaro.
//
//Assuming that inner = <v,w> (inner product of v and w),
//v and w are considered orthogonal if
//  |inner| < 100 * ||v||_2 * ||w||_2 * epsilon

  magnitude vnorm = std::sqrt(dot(v,v,NULL));
  magnitude wnorm = std::sqrt(dot(w,w,NULL));
  return std::abs(inner) <= 100*vnorm*wnorm*std::numeric_limits<magnitude>::epsilon();
}

template<typename OperatorType,
         typename VectorType,
         typename Matvec>
void
cg_solve(OperatorType& A,
         const VectorType& b,
         VectorType& x,
         Matvec matvec,
         typename OperatorType::LocalOrdinalType max_iter,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance,
         typename OperatorType::LocalOrdinalType& num_iters,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr,
         timer_type* my_cg_times)
{
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
  typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
  typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  timer_type t0 = 0, tWAXPY = 0, tDOT = 0, tMATVEC = 0, tMATVECDOT = 0;
  timer_type total_time = mytimer();

  int myproc = 0, world_size;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  if (!A.has_local_indices) {
    std::cerr << "miniFE::cg_solve ERROR, A.has_local_indices is false, needs to be true. This probably means "
       << "miniFE::make_local_matrix(A) was not called prior to calling miniFE::cg_solve."
       << std::endl;
    return;
  }

  size_t nrows = A.rows.size();
  LocalOrdinalType ncols = A.num_cols;

  nvtxRangeId_t r1=nvtxRangeStartA("Allocation of Temporary Vectors");
  VectorType r(b.startIndex, nrows);
  VectorType p(0, ncols);
  VectorType Ap(b.startIndex, nrows);
  nvtxRangeEnd(r1);

#ifdef HAVE_MPI
#ifndef GPUDIRECT
  //TODO move outside?
  /*
  cudaHostRegister(&p.coefs[0],ncols*sizeof(typename VectorType::ScalarType),0);
  cudaCheckError();
  if(A.send_buffer.size()>0) cudaHostRegister(&A.send_buffer[0],A.send_buffer.size()*sizeof(typename VectorType::ScalarType),0);
  cudaCheckError();
  */
#endif
#endif

  normr = 0;
  magnitude_type rtrans = 0;
  magnitude_type oldrtrans = 0;

  LocalOrdinalType print_freq = max_iter/10;
  if (print_freq>50) print_freq = 50;
  if (print_freq<1)  print_freq = 1;

  ScalarType one = 1.0;
  ScalarType zero = 0.0;

  TICK(); waxpby(one, x, zero, x, p); TOCK(tWAXPY);

  TICK();
  matvec(A, p, Ap, NULL, NULL);
  TOCK(tMATVEC);

  TICK(); waxpby(one, b, -one, Ap, r); TOCK(tWAXPY);

  TICK(); rtrans = dot(r, r, NULL); TOCK(tDOT);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "brkdown_tol = " << brkdown_tol << std::endl;
#endif

#define TIME_COUNT 9
  // For timing
  // 0: Iteration
  // 1-2: Count 1, MPI_Irecv, MPI_Isend
  // 3-4: Count 200, MPI_Irecv, MPI_Isend
  // 5-6: Count 40000, MPI_Irecv, MPI_Isend
  // 7: MPI_Waitall
  // 8: MPI_Allreduce
  double times[TIME_COUNT] = {0.0};
  double acc_times[TIME_COUNT] = {0.0};
  double acc_times_sum[TIME_COUNT];
  double acc_times_max[TIME_COUNT];
  int call_counts[3] = {0}; // 0: Count 1, 1: Count 200, 2: Count 40000
  double iter_start_time;

  for(LocalOrdinalType k=1; k <= max_iter && normr > tolerance; ++k) {
    iter_start_time = MPI_Wtime();

    if (k == 1) {
      TICK(); waxpby(one, r, zero, r, p); TOCK(tWAXPY);
    }
    else {
      oldrtrans = rtrans;
      TICK(); rtrans = dot(r, r, times); TOCK(tDOT);
      acc_times[8] += times[8];
      magnitude_type beta = rtrans/oldrtrans;
      TICK(); waxpby(one, r, beta, p, p); TOCK(tWAXPY);
    }

    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    TICK(); matvec(A, p, Ap, times, call_counts); TOCK(tMATVEC);

    TICK(); p_ap_dot = dot(Ap, p, times); TOCK(tDOT);
    acc_times[8] += times[8];

#ifdef MINIFE_DEBUG
    os << "iter " << k << ", p_ap_dot = " << p_ap_dot;
    os.flush();
#endif
    //TODO remove false below
    if (false && p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 || breakdown(p_ap_dot, Ap, p)) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
#ifdef MINIFE_DEBUG
        os << "ERROR, numerical breakdown!"<<std::endl;
#endif
        //update the timers before jumping out.
        my_cg_times[WAXPY] = tWAXPY;
        my_cg_times[DOT] = tDOT;
        my_cg_times[MATVEC] = tMATVEC;
        my_cg_times[TOTAL] = mytimer() - total_time;
        return;
      }
      else brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans/p_ap_dot;
#ifdef MINIFE_DEBUG
    os << ", rtrans = " << rtrans << ", alpha = " << alpha << std::endl;
#endif

    TICK(); waxpby(one, x, alpha, p, x);
            waxpby(one, r, -alpha, Ap, r); TOCK(tWAXPY);

    // Compute iteration time
    times[0] = MPI_Wtime() - iter_start_time;
    acc_times[0] += times[0];

    // FIXME WRONG!!! Accumulate times
    //for (int i = 0; i < TIME_COUNT; i++) acc_times[i] += times[i];

    num_iters = k;
  }

  // Get average iteration time
  acc_times[0] /= num_iters;

  // Get average time per iteration for each MPI call
  for (int i = 1; i <= 2; i++) { // Count 1
    if (call_counts[0] > 0) acc_times[i] /= call_counts[0];
    else acc_times[i] = 0.0;
  }
  for (int i = 3; i <= 4; i++) { // Count 200
    if (call_counts[1] > 0) acc_times[i] /= call_counts[1];
    else acc_times[i] = 0.0;
  }
  for (int i = 5; i <= 6; i++) { // Count 40000
    if (call_counts[2] > 0) acc_times[i] /= call_counts[2];
    else acc_times[i] = 0.0;
  }

  // Get average time per iteration for MPI_Waitall and MPI_Allreduce
  acc_times[7] /= num_iters;
  acc_times[8] /= 2*num_iters-1;

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Reduce(acc_times, acc_times_sum, TIME_COUNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(acc_times, acc_times_max, TIME_COUNT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (myproc == 0) {
    // Get average times over all MPI ranks
    for (int i = 0; i < TIME_COUNT; i++) {
      acc_times_sum[i] /= world_size;
    }

    printf("[Average] Iteration: %.6lf us\n", acc_times_sum[0] * 1000000);
    printf("[Average] Count 1, MPI_Irecv: %.6lf us, MPI_Isend: %.6lf us\n",
        acc_times_sum[1] * 1000000, acc_times_sum[2] * 1000000);
    printf("[Average] Count 200, MPI_Irecv: %.6lf us, MPI_Isend: %.6lf us\n",
        acc_times_sum[3] * 1000000, acc_times_sum[4] * 1000000);
    printf("[Average] Count 40000, MPI_Irecv: %.6lf us, MPI_Isend: %.6lf us\n",
        acc_times_sum[5] * 1000000, acc_times_sum[6] * 1000000);
    printf("[Average] MPI_Waitall: %.6lf us, MPI_Allreduce: %.6lf us\n",
        acc_times_sum[7] * 1000000, acc_times_sum[8] * 1000000);
    printf("[Max] Iteration: %.6lf us\n", acc_times_max[0] * 1000000);
    printf("[Max] Count 1, MPI_Irecv: %.6lf us, MPI_Isend: %.6lf us\n",
        acc_times_max[1] * 1000000, acc_times_max[2] * 1000000);
    printf("[Max] Count 200, MPI_Irecv: %.6lf us, MPI_Isend: %.6lf us\n",
        acc_times_max[3] * 1000000, acc_times_max[4] * 1000000);
    printf("[Max] Count 40000, MPI_Irecv: %.6lf us, MPI_Isend: %.6lf us\n",
        acc_times_max[5] * 1000000, acc_times_max[6] * 1000000);
    printf("[Max] MPI_Waitall: %.6lf us, MPI_Allreduce: %.6lf us\n",
        acc_times_max[7] * 1000000, acc_times_max[8] * 1000000);
  }
  
#ifdef HAVE_MPI
#ifndef GPUDIRECT
  //TODO move outside?
  cudaHostUnregister(&p.coefs[0]);
  cudaCheckError();
  if(A.send_buffer.size()>0) cudaHostUnregister(&A.send_buffer[0]);
  cudaCheckError();
#endif
#endif

  my_cg_times[WAXPY] = tWAXPY;
  my_cg_times[DOT] = tDOT;
  my_cg_times[MATVEC] = tMATVEC;
  my_cg_times[MATVECDOT] = tMATVECDOT;
  my_cg_times[TOTAL] = mytimer() - total_time;
}

}//namespace miniFE

#endif

