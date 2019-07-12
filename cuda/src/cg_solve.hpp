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

  magnitude vnorm = std::sqrt(dot(v,v,-1,NULL));
  magnitude wnorm = std::sqrt(dot(w,w,-1,NULL));
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

  int myproc = 0;
#ifdef HAVE_MPI
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
  cudaHostRegister(&p.coefs[0],ncols*sizeof(typename VectorType::ScalarType),0);
  cudaCheckError();
  if(A.send_buffer.size()>0) cudaHostRegister(&A.send_buffer[0],A.send_buffer.size()*sizeof(typename VectorType::ScalarType),0);
  cudaCheckError();
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

  TICK(); waxpby(one, x, zero, x, p, -1); TOCK(tWAXPY);

  TICK();
  matvec(A, p, Ap, NULL);
  TOCK(tMATVEC);

  TICK(); waxpby(one, b, -one, Ap, r, -1); TOCK(tWAXPY);

  TICK(); rtrans = dot(r, r, -1, NULL); TOCK(tDOT);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "brkdown_tol = " << brkdown_tol << std::endl;
#endif

#ifdef TIME_BREAKDOWN
  double dot_times[8] = {0.0};
  double waxpby_times[3] = {0.0};
  double matvec_times[5] = {0.0};
#endif

  for(LocalOrdinalType k=1; k <= max_iter && normr > tolerance; ++k) {
    double dot_mpi_times[2];
    double matvec_mpi_time;
    if (k == 1) {
      TICK(); waxpby(one, r, zero, r, p, 1); TOCK(tWAXPY);
    }
    else {
      oldrtrans = rtrans;
      TICK(); rtrans = dot(r, r, 1, dot_mpi_times); TOCK(tDOT);
      magnitude_type beta = rtrans/oldrtrans;
      TICK(); waxpby(one, r, beta, p, p, 1); TOCK(tWAXPY);
    }

    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    TICK(); matvec(A, p, Ap, &matvec_mpi_time); TOCK(tMATVEC);

    TICK(); p_ap_dot = dot(Ap, p, 2, dot_mpi_times); TOCK(tDOT);

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

    TICK(); waxpby(one, x, alpha, p, x, 2);
            waxpby(one, r, -alpha, Ap, r, 3); TOCK(tWAXPY);

#ifdef TIME_BREAKDOWN
    cudaDeviceSynchronize();
    if (k > 1) {
      // Compute CUDA times
      float dot_times_cur[8];
      float waxpby_times_cur[3];
      float matvec_times_cur[5];
      cudaEventElapsedTime(&dot_times_cur[0], CudaManager::et[0], CudaManager::et[1]);
      cudaEventElapsedTime(&dot_times_cur[1], CudaManager::et[1], CudaManager::et[2]);
      cudaEventElapsedTime(&dot_times_cur[2], CudaManager::et[2], CudaManager::et[3]);
      // dot_times_cur[3] is MPI_Allreduce time
      cudaEventElapsedTime(&waxpby_times_cur[0], CudaManager::et[4], CudaManager::et[5]);
      cudaEventElapsedTime(&matvec_times_cur[0], CudaManager::et[6], CudaManager::et[7]);
      cudaEventElapsedTime(&matvec_times_cur[1], CudaManager::et[7], CudaManager::et[8]);
      cudaEventElapsedTime(&matvec_times_cur[2], CudaManager::et[9], CudaManager::et[10]);
      // matvec_times_cur[3] is MPI comm time
      cudaEventElapsedTime(&matvec_times_cur[4], CudaManager::et[11], CudaManager::et[12]);
      cudaEventElapsedTime(&dot_times_cur[4], CudaManager::et[13], CudaManager::et[14]);
      cudaEventElapsedTime(&dot_times_cur[5], CudaManager::et[14], CudaManager::et[15]);
      cudaEventElapsedTime(&dot_times_cur[6], CudaManager::et[15], CudaManager::et[16]);
      // dot_times_cur[7] is MPI_Allreduce time
      cudaEventElapsedTime(&waxpby_times_cur[1], CudaManager::et[17], CudaManager::et[18]);
      cudaEventElapsedTime(&waxpby_times_cur[2], CudaManager::et[19], CudaManager::et[20]);

      // Print times
      if (myproc == 0) {
        printf("[%d][DOT-1] dot_kernel: %.6f, dot_final_reduce_kernel: %.6f, D2H memcpy: %.6f, MPI_Allreduce: %.6lf\n", k, dot_times_cur[0], dot_times_cur[1], dot_times_cur[2], dot_mpi_times[0]);
        printf("[%d][WAXPBY-1] waxpby_kernel: %.6f\n", k, waxpby_times_cur[0]);
        printf("[%d][MATVEC] copyElementsToBuffer: %.6f, D2H memcpy: %.6f, int kernel: %.6f, MPI comm: %.6lf, ext kernel: %.6f\n", k, matvec_times_cur[0], matvec_times_cur[1], matvec_times_cur[2], matvec_mpi_time, matvec_times_cur[4]);
        printf("[%d][DOT-2] dot_kernel: %.6f, dot_final_reduce_kernel: %.6f, D2H memcpy: %.6f, MPI_Allreduce: %.6lf\n", k, dot_times_cur[4], dot_times_cur[5], dot_times_cur[6], dot_mpi_times[1]);
        printf("[%d][WAXPBY-2] waxpby_kernel: %.6f\n", k, waxpby_times_cur[1]);
        printf("[%d][WAXPBY-3] waxpby_kernel: %.6f\n", k, waxpby_times_cur[2]);
      }

      // Accumulate times
      for (int i = 0; i <= 2; i++) dot_times[i] += dot_times_cur[i];
      dot_times[3] += dot_mpi_times[0];
      for (int i = 4; i <= 6; i++) dot_times[i] += dot_times_cur[i];
      dot_times[7] += dot_mpi_times[1];
      for (int i = 0; i <= 2; i++) waxpby_times[i] += waxpby_times_cur[i];
      for (int i = 0; i <= 2; i++) matvec_times[i] += matvec_times_cur[i];
      matvec_times[3] += matvec_mpi_time;
      matvec_times[4] += matvec_times_cur[4];
    }
#endif

    num_iters = k;
  }

#ifdef TIME_BREAKDOWN
  // Print accumulated times
  if (myproc == 0) {
    printf("!!! [DOT-1] dot_kernel: %.6lf/%.6lf, dot_final_reduce_kernel: %.6lf/%.6lf, D2H memcpy: %.6lf/%.6lf, MPI_Allreduce: %.6lf/%.6lf\n", dot_times[0], dot_times[0]/(num_iters-1), dot_times[1], dot_times[1]/(num_iters-1), dot_times[2], dot_times[2]/(num_iters-1), dot_times[3], dot_times[3]/(num_iters-1));
    printf("!!! [WAXPBY-1] waxpby_kernel: %.6lf/%.6lf\n", waxpby_times[0], waxpby_times[0]/(num_iters-1));
    printf("!!! [MATVEC] copyElementsToBuffer: %.6lf/%.6lf, D2H memcpy: %.6lf/%.6lf, int kernel: %.6lf/%.6lf, MPI comm: %.6lf/%.6lf, ext kernel: %.6lf/%.6lf\n", matvec_times[0], matvec_times[0]/(num_iters-1), matvec_times[1], matvec_times[1]/(num_iters-1), matvec_times[2], matvec_times[2]/(num_iters-1), matvec_times[3], matvec_times[3]/(num_iters-1), matvec_times[4], matvec_times[4]/(num_iters-1));
    printf("!!! [DOT-2] dot_kernel: %.6lf/%.6lf, dot_final_reduce_kernel: %.6lf/%.6lf, D2H memcpy: %.6lf/%.6lf, MPI_Allreduce: %.6lf/%.6lf\n", dot_times[4], dot_times[4]/(num_iters-1), dot_times[5], dot_times[5]/(num_iters-1), dot_times[6], dot_times[6]/(num_iters-1), dot_times[7], dot_times[7]/(num_iters-1));
    printf("!!! [WAXPBY-2] waxpby_kernel: %.6lf/%.6lf\n", waxpby_times[1], waxpby_times[1]/(num_iters-1));
    printf("!!! [WAXPBY-3] waxpby_kernel: %.6lf/%.6lf\n", waxpby_times[2], waxpby_times[2]/(num_iters-1));
  }
#endif
  
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

