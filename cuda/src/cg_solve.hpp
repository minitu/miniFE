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

#define N_TIMER 11
#define N_DUR 5

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

  magnitude vnorm = std::sqrt(dot(v,v,NULL,-1));
  magnitude wnorm = std::sqrt(dot(w,w,NULL,-1));
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

  int rank = 0, world_size;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
  matvec(A, p, Ap, NULL);
  TOCK(tMATVEC);

  TICK(); waxpby(one, b, -one, Ap, r); TOCK(tWAXPY);

  TICK(); rtrans = dot(r, r, NULL, -1); TOCK(tDOT);

  normr = std::sqrt(rtrans);

  if (rank == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "brkdown_tol = " << brkdown_tol << std::endl;
#endif

  double times[N_TIMER];
  double durs[N_DUR];
  double durs_min[N_DUR];
  double durs_sum[N_DUR];
  for (int i = 0; i < N_DUR; i++) {
    durs_min[i] = std::numeric_limits<double>::max();
    durs_sum[i] = 0;
  }
  double durs_global[N_DUR*world_size];

  // Main iteration loop
  for(LocalOrdinalType k=1; k <= max_iter && normr > tolerance; ++k) {
    for (int i = 0; i < N_TIMER; i++) {
      times[i] = 0;
    }
    times[0] = MPI_Wtime();

    if (k == 1) {
      TICK(); waxpby(one, r, zero, r, p); TOCK(tWAXPY);
    }
    else {
      oldrtrans = rtrans;
      TICK(); rtrans = dot(r, r, times, 0); TOCK(tDOT);
      magnitude_type beta = rtrans/oldrtrans;
      TICK(); waxpby(one, r, beta, p, p); TOCK(tWAXPY);
    }

    normr = std::sqrt(rtrans);

    if (rank == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    TICK(); matvec(A, p, Ap, times); TOCK(tMATVEC);

    TICK(); p_ap_dot = dot(Ap, p, times, 1); TOCK(tDOT);

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

    times[10] = MPI_Wtime();

#ifdef MEASURE_TIME
    // Compute durations
    durs[0] = times[3] - times[2]; // allreduce 1
    durs[1] = times[6] - times[5]; // halo
    durs[2] = times[9] - times[8]; // allreduce 2
    durs[3] = (times[1] - times[0]) + (times[4] - times[3]) + (times[7] - times[6])
      + (times[10] - times[9]); // overhead
    durs[4] = (times[10] - times[0]) - (times[2] - times[1]) - (times[5] - times[4])
      - (times[8] - times[7]); // iteration time (minus barrier times)

    // Calculate min and sum but don't include 1st iteration
    if (k > 1) {
      for (int i = 0; i < N_DUR; i++) {
        if (durs[i] < durs_min[i]) {
          durs_min[i] = durs[i];
        }
        durs_sum[i] += durs[i];
      }
    }

    // Print per-iteration times
    MPI_Gather(durs, N_DUR, MPI_DOUBLE, durs_global, N_DUR, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      for (int j = 0; j < world_size; j++) {
        // Convert to us
        for (int i = 0; i < N_DUR; i++) {
          durs_global[N_DUR*j+i] *= 1000000;
        }
        printf("[%03d,%03d] Allreduce 1: %.3lf, Halo: %.3lf, Allreduce 2: %.3lf, "
            "Overhead (inc. kernels): %.3lf, Iteration: %.3lf\n", k, j,
            durs_global[N_DUR*j], durs_global[N_DUR*j+1], durs_global[N_DUR*j+2],
            durs_global[N_DUR*j+3], durs_global[N_DUR*j+4]);
      }
    }
#endif

    num_iters = k;
  }

#ifdef MEASURE_TIME
  // Print average times
  MPI_Gather(durs_sum, N_DUR, MPI_DOUBLE, durs_global, N_DUR, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      // Compute average and convert to us
      for (int i = 0; i < N_DUR; i++) {
        durs_global[N_DUR*j+i] /= (num_iters-1);
        durs_global[N_DUR*j+i] *= 1000000;
      }

      printf("[average,%03d] Allreduce 1: %.3lf, Halo: %.3lf, Allreduce 2: %.3lf, "
          "Overhead (inc. kernels): %.3lf, Iteration: %.3lf\n", j,
          durs_global[N_DUR*j], durs_global[N_DUR*j+1], durs_global[N_DUR*j+2],
          durs_global[N_DUR*j+3], durs_global[N_DUR*j+4]);
    }
  }

  // Print minimum times
  MPI_Gather(durs_min, N_DUR, MPI_DOUBLE, durs_global, N_DUR, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      // Convert to us
      for (int i = 0; i < N_DUR; i++) {
        durs_global[N_DUR*j+i] *= 1000000;
      }

      printf("[minimum,%03d] Allreduce 1: %.3lf, Halo: %.3lf, Allreduce 2: %.3lf, "
          "Overhead (inc. kernels): %.3lf, Iteration: %.3lf\n", j,
          durs_global[N_DUR*j], durs_global[N_DUR*j+1], durs_global[N_DUR*j+2],
          durs_global[N_DUR*j+3], durs_global[N_DUR*j+4]);
    }
  }

  // Print times of rank with maximum halo and maximum allreduce 1
  if (rank == 0) {
    double max_halo = 0;
    double max_allreduce = 0;
    int max_halo_rank = 0;
    int max_allreduce_rank = 0;

    for (int j = 0; j < world_size; j++) {
      if (durs_global[N_DUR*j+1] > max_halo) {
        max_halo = durs_global[N_DUR*j+1];
        max_halo_rank = j;
      }
      if (durs_global[N_DUR*j] > max_allreduce) {
        max_allreduce = durs_global[N_DUR*j];
        max_allreduce_rank = j;
      }
    }

    printf("[max halo] Allreduce 1: %.3lf, Halo: %.3lf, Allreduce 2: %.3lf, "
        "Overhead (inc. kernels): %.3lf, Iteration: %.3lf\n",
        durs_global[N_DUR*max_halo_rank], durs_global[N_DUR*max_halo_rank+1],
        durs_global[N_DUR*max_halo_rank+2], durs_global[N_DUR*max_halo_rank+3],
        durs_global[N_DUR*max_halo_rank+4]);

    printf("[max allreduce] Allreduce 1: %.3lf, Halo: %.3lf, Allreduce 2: %.3lf, "
        "Overhead (inc. kernels): %.3lf, Iteration: %.3lf\n",
        durs_global[N_DUR*max_allreduce_rank], durs_global[N_DUR*max_allreduce_rank+1],
        durs_global[N_DUR*max_allreduce_rank+2], durs_global[N_DUR*max_allreduce_rank+3],
        durs_global[N_DUR*max_allreduce_rank+4]);
  }
#endif
  
#ifdef HAVE_MPI
#ifndef GPUDIRECT
  //TODO move outside?
  /*
  cudaHostUnregister(&p.coefs[0]);
  cudaCheckError();
  if(A.send_buffer.size()>0) cudaHostUnregister(&A.send_buffer[0]);
  cudaCheckError();
  */
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

