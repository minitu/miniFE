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

#ifdef DUMPI_TRACE
#include <dumpi/libdumpi/libdumpi.h>
#endif

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

  TICK(); rtrans = dot(r, r, -1, NULL); TOCK(tDOT);

  normr = std::sqrt(rtrans);

  if (rank == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();

#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "brkdown_tol = " << brkdown_tol << std::endl;
#endif

#ifdef DUMPI_TRACE
  // Turn on DUMPI tracing
  libdumpi_enable_profiling();
#endif

  double* iter_times;
  double* dot1_times; // MPI_Barrier, MPI_Allreduce, Dot total
  double* waxpby1_times;
  double* matvec_times; // MPI_Barrier, MPI_Irecvs, MPI_Isends, MPI_Waitall, GPU kernel, Matvec total
  double* dot2_times;
  double* waxpby2_times;
  double* waxpby3_times;

  iter_times = (double*)malloc(sizeof(double) * max_iter);
  dot1_times = (double*)malloc(sizeof(double) * max_iter * 3);
  waxpby1_times = (double*)malloc(sizeof(double) * max_iter);
  matvec_times = (double*)malloc(sizeof(double) * max_iter * 6);
  dot2_times = (double*)malloc(sizeof(double) * max_iter * 3);
  waxpby2_times = (double*)malloc(sizeof(double) * max_iter);
  waxpby3_times = (double*)malloc(sizeof(double) * max_iter);

  // Main iteration loop
  for(LocalOrdinalType k=1; k <= max_iter && normr > tolerance; ++k) {
    iter_times[k-1] = MPI_Wtime();

    if (k == 1) {
      dot1_times[(k-1)*3] = 0;
      dot1_times[(k-1)*3+1] = 0;
      dot1_times[(k-1)*3+2] = 0;
      waxpby1_times[k-1] = MPI_Wtime();
      TICK(); waxpby(one, r, zero, r, p); TOCK(tWAXPY);
      waxpby1_times[k-1] = MPI_Wtime() - waxpby1_times[k-1];
    }
    else {
      oldrtrans = rtrans;
      dot1_times[(k-1)*3+2] = MPI_Wtime();
      TICK(); rtrans = dot(r, r, 0, &dot1_times[(k-1)*3]); TOCK(tDOT);
      dot1_times[(k-1)*3+2] = MPI_Wtime() - dot1_times[(k-1)*3+2];
      magnitude_type beta = rtrans/oldrtrans;
      waxpby1_times[k-1] = MPI_Wtime();
      TICK(); waxpby(one, r, beta, p, p); TOCK(tWAXPY);
      waxpby1_times[k-1] = MPI_Wtime() - waxpby1_times[k-1];
    }

    normr = std::sqrt(rtrans);

    if (rank == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    matvec_times[(k-1)*6+5] = MPI_Wtime();
    TICK(); matvec(A, p, Ap, &matvec_times[(k-1)*6]); TOCK(tMATVEC);
    matvec_times[(k-1)*6+5] = MPI_Wtime() - matvec_times[(k-1)*6+5];

    dot2_times[(k-1)*3+2] = MPI_Wtime();
    TICK(); p_ap_dot = dot(Ap, p, 1, &dot2_times[(k-1)*3]); TOCK(tDOT);
    dot2_times[(k-1)*3+2] = MPI_Wtime() - dot2_times[(k-1)*3+2];

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

    TICK();
    waxpby2_times[k-1] = MPI_Wtime();
    waxpby(one, x, alpha, p, x);
    waxpby2_times[k-1] = MPI_Wtime() - waxpby2_times[k-1];
    waxpby3_times[k-1] = MPI_Wtime();
    waxpby(one, r, -alpha, Ap, r);
    waxpby3_times[k-1] = MPI_Wtime() - waxpby3_times[k-1];
    TOCK(tWAXPY);

    iter_times[k-1] = MPI_Wtime() - iter_times[k-1];
    num_iters = k;
  }

#ifdef DUMPI_TRACE
  // Turn off DUMPI tracing
  libdumpi_disable_profiling();
#endif

#ifdef MEASURE_TIME
  double* iter_times_global;
  double* dot1_times_global;
  double* waxpby1_times_global;
  double* matvec_times_global;
  double* dot2_times_global;
  double* waxpby2_times_global;
  double* waxpby3_times_global;

  if (rank == 0) {
    iter_times_global = (double*)malloc(sizeof(double) * world_size * max_iter);
    dot1_times_global = (double*)malloc(sizeof(double) * world_size * max_iter * 3);
    waxpby1_times_global = (double*)malloc(sizeof(double) * world_size * max_iter);
    matvec_times_global = (double*)malloc(sizeof(double) * world_size * max_iter * 6);
    dot2_times_global = (double*)malloc(sizeof(double) * world_size * max_iter * 3);
    waxpby2_times_global = (double*)malloc(sizeof(double) * world_size * max_iter);
    waxpby3_times_global = (double*)malloc(sizeof(double) * world_size * max_iter);
  }

  MPI_Gather(iter_times, max_iter, MPI_DOUBLE, iter_times_global, max_iter, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(dot1_times, max_iter * 3, MPI_DOUBLE, dot1_times_global, max_iter * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(waxpby1_times, max_iter, MPI_DOUBLE, waxpby1_times_global, max_iter, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(matvec_times, max_iter * 6, MPI_DOUBLE, matvec_times_global, max_iter * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(dot2_times, max_iter * 3, MPI_DOUBLE, dot2_times_global, max_iter * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(waxpby2_times, max_iter, MPI_DOUBLE, waxpby2_times_global, max_iter, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(waxpby3_times, max_iter, MPI_DOUBLE, waxpby3_times_global, max_iter, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double waitall_max = 0;
    double waitall_max_values[16];
    double waitall_max_rank;

    for (int r = 0; r < world_size; r++) {
      /*
      for (int i = 0; i < max_iter; i++) {
        printf("[Iteration %d]\n", i);
        printf("DOT1: %.3lf\n", dot1_times_global[max_iter*r+i] * 1000000);
        printf("WAXPBY1: %.3lf\n", waxpby1_times_global[max_iter*r+i] * 1000000);
        printf("MATVEC: %.3lf (Irecv %.3lf -> Isend %.3lf -> Waitall %.3lf)\n",
            matvec_times_global[(max_iter*r+i)*4+3] * 1000000, matvec_times_global[(max_iter*r+i)*4] * 1000000,
            matvec_times_global[(max_iter*r+i)*4+1] * 1000000, matvec_times_global[(max_iter*r+i)*4+2] * 1000000);
        printf("DOT2: %.3lf\n", dot2_times_global[max_iter*r+i] * 1000000);
        printf("WAXPBY2: %.3lf\n", waxpby2_times_global[max_iter*r+i] * 1000000);
        printf("WAXPBY3: %.3lf\n", waxpby3_times_global[max_iter*r+i] * 1000000);
        printf("Iteration: %.3lf\n", iter_times_global[max_iter*r+i] * 1000000);
      }
      */
      double dot1_averages[3] = {0, 0, 0};
      double waxpby1_average = 0;
      double matvec_averages[6] = {0, 0, 0, 0, 0, 0};
      double dot2_averages[3] = {0, 0, 0};
      double waxpby2_average = 0;
      double waxpby3_average = 0;
      double iter_average = 0;

      // Exclude 1st iteration from average
      for (int i = 1; i < max_iter; i++) {
        for (int j = 0; j < 3; j++) {
          dot1_averages[j] += dot1_times_global[(max_iter*r+i)*3+j];
        }
        waxpby1_average += waxpby1_times_global[max_iter*r+i];
        for (int j = 0; j < 6; j++) {
          matvec_averages[j] += matvec_times_global[(max_iter*r+i)*6+j];
        }
        for (int j = 0; j < 3; j++) {
          dot2_averages[j] += dot2_times_global[(max_iter*r+i)*3+j];
        }
        waxpby2_average += waxpby2_times_global[max_iter*r+i];
        waxpby3_average += waxpby3_times_global[max_iter*r+i];
        iter_average += iter_times_global[max_iter*r+i];
      }
      for (int j = 0; j < 3; j++) {
        dot1_averages[j] /= (max_iter-1);
      }
      waxpby1_average /= (max_iter-1);
      for (int j = 0; j < 6; j++) {
        matvec_averages[j] /= (max_iter-1);
      }
      for (int j = 0; j < 3; j++) {
        dot2_averages[j] /= (max_iter-1);
      }
      waxpby2_average /= (max_iter-1);
      waxpby3_average /= (max_iter-1);
      iter_average /= (max_iter-1);

      printf("[Rank %4d] Dot1: %.3lf (Barrier: %.3lf -> Allreduce: %.3lf), Waxpby1: %.3lf, "
          "Matvec: %.3lf (Barrier: %.3lf -> Irecv %.3lf -> Isend %.3lf -> Waitall %.3lf -> "
          "GPU: %.3lf), Dot2: %.3lf (Barrier: %.3lf -> Allreduce: %.3lf), Waxpby2: %.3lf, "
          "Waxpby3: %.3lf, Iter: %.3lf\n",
          r, dot1_averages[2] * 1000000, dot1_averages[0] * 1000000, dot1_averages[1] * 1000000,
          waxpby1_average * 1000000, matvec_averages[5] * 1000000, matvec_averages[0] * 1000000,
          matvec_averages[1] * 1000000, matvec_averages[2] * 1000000, matvec_averages[3] * 1000000,
          matvec_averages[4] * 1000000, dot2_averages[2] * 1000000, dot2_averages[0] * 1000000,
          dot2_averages[1] * 1000000, waxpby2_average * 1000000, waxpby3_average * 1000000,
          iter_average * 1000000);

      // Find rank with maximum waitall time
      if (matvec_averages[3] > waitall_max) {
        waitall_max = matvec_averages[3];
        waitall_max_rank = r;
        waitall_max_values[0] = dot1_averages[2];
        waitall_max_values[1] = dot1_averages[0];
        waitall_max_values[2] = dot1_averages[1];
        waitall_max_values[3] = waxpby1_average;
        waitall_max_values[4] = matvec_averages[5];
        waitall_max_values[5] = matvec_averages[0];
        waitall_max_values[6] = matvec_averages[1];
        waitall_max_values[7] = matvec_averages[2];
        waitall_max_values[8] = matvec_averages[3];
        waitall_max_values[9] = matvec_averages[4];
        waitall_max_values[10] = dot2_averages[2];
        waitall_max_values[11] = dot2_averages[0];
        waitall_max_values[12] = dot2_averages[1];
        waitall_max_values[13] = waxpby2_average;
        waitall_max_values[14] = waxpby3_average;
        waitall_max_values[15] = iter_average;
      }
    }

    printf("[Max waitall, Rank %4d] Dot1: %.3lf (Barrier: %.3lf -> Allreduce: %.3lf), Waxpby1: %.3lf, "
        "Matvec: %.3lf (Barrier: %.3lf -> Irecv %.3lf -> Isend %.3lf -> Waitall %.3lf -> "
        "GPU: %.3lf), Dot2: %.3lf (Barrier: %.3lf -> Allreduce: %.3lf), Waxpby2: %.3lf, "
        "Waxpby3: %.3lf, Iter: %.3lf\n",
        waitall_max_rank, waitall_max_values[0] * 1000000, waitall_max_values[1] * 1000000,
        waitall_max_values[2] * 1000000, waitall_max_values[3] * 1000000, waitall_max_values[4] * 1000000,
        waitall_max_values[5] * 1000000, waitall_max_values[6] * 1000000, waitall_max_values[7] * 1000000,
        waitall_max_values[8] * 1000000, waitall_max_values[9] * 1000000, waitall_max_values[10] * 1000000,
        waitall_max_values[11] * 1000000, waitall_max_values[12] * 1000000, waitall_max_values[13] * 1000000,
        waitall_max_values[14] * 1000000, waitall_max_values[15] * 1000000);
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

