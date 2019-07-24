#ifndef _exchange_externals_hpp_
#define _exchange_externals_hpp_

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

#include <cstdlib>
#include <iostream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <outstream.hpp>
#include <fstream>

#include <TypeTraits.hpp>

namespace miniFE {

template<typename Scalar, typename Index> 
  __global__ void copyElementsToBuffer(Scalar *src, Scalar *dst, Index *indices, int N) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    int idx=indices[i];
    dst[i]=__ldg(src+idx);
  }
}

template<typename MatrixType,
         typename VectorType>
void
exchange_externals(MatrixType& A,
                   VectorType& x, double* times, int* call_counts)
{
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "entering exchange_externals\n";
#endif

  int numprocs = 1, myproc;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  if (numprocs < 2) return;
  
  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  // Extract Matrix pieces

  int local_nrow = A.rows.size();
  int num_neighbors = A.neighbors.size();
  const std::vector<LocalOrdinal>& recv_length = A.recv_length;
  const std::vector<LocalOrdinal>& send_length = A.send_length;
  const std::vector<int>& neighbors = A.neighbors;
  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  std::vector<MPI_Request>& request = A.request;

  //
  // Externals are at end of locals
  //
  
  //
  // Fill up send buffer
  //

  int BLOCK_SIZE=256;
  int BLOCKS=min((int)(A.d_elements_to_send.size()+BLOCK_SIZE-1)/BLOCK_SIZE,2048);

  copyElementsToBuffer<<<BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(thrust::raw_pointer_cast(&x.d_coefs[0]),
                                     thrust::raw_pointer_cast(&A.d_send_buffer[0]), 
                                     thrust::raw_pointer_cast(&A.d_elements_to_send[0]),
                                     A.d_elements_to_send.size());
  cudaCheckError();

#ifndef GPUDIRECT
  std::vector<Scalar>& send_buffer = A.send_buffer;
  //wait for packing to finish
  cudaMemcpyAsync(&send_buffer[0],thrust::raw_pointer_cast(&A.d_send_buffer[0]),sizeof(Scalar)*A.d_elements_to_send.size(),cudaMemcpyDeviceToHost,CudaManager::s1);
  cudaCheckError();
#endif
  cudaEventRecord(CudaManager::e1,CudaManager::s1);

#ifdef GPUDIRECT
  Scalar * x_external = thrust::raw_pointer_cast(&x.d_coefs[local_nrow]);
#else
  std::vector<Scalar>& x_coefs = x.coefs;
  Scalar* x_external = &(x_coefs[local_nrow]);
#endif

  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

  double mpi_start_time;
#if 0
  /* Internal pingpong test */
  static bool start = false;

  if (!start) {
    start = true;
    MPI_Request requests[2];
    MPI_Status waits[2];

    std::vector<int> double_counts = {1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768};
    int double_count = 1024;
    while (double_count <= 1536) {
      double_counts.push_back(double_count);
      double_count += 32;
    }
    while (double_count <= 50000) {
      double_counts.push_back(double_count);
      double_count += 512;
    }

    double* recv_buf = (double*)std::malloc(sizeof(double) * 1048576);
    double* send_buf = (double*)std::malloc(sizeof(double) * 1048576);
    int types = double_counts.size();
    double* acc_times = (double*)std::malloc(sizeof(double) * types * 3);
    double* global_acc_times = (double*)std::malloc(sizeof(double) * types * 3);

    for (int i = 0; i < types; i++) {
      int count = double_counts[i];
      for (int j = 0; j < 3; j++) acc_times[3*i+j] = global_acc_times[3*i+j] = 0.0;

      MPI_Barrier(MPI_COMM_WORLD);

      for (int j = 0; j < 1000; j++) {
        mpi_start_time = MPI_Wtime();
        MPI_Irecv(recv_buf, count, MPI_DOUBLE, 1-myproc, MPI_MY_TAG, MPI_COMM_WORLD, &requests[0]);
        acc_times[3*i] += MPI_Wtime() - mpi_start_time;

        mpi_start_time = MPI_Wtime();
        MPI_Isend(send_buf, count, MPI_DOUBLE, 1-myproc, MPI_MY_TAG, MPI_COMM_WORLD, &requests[1]);
        acc_times[3*i+1] += MPI_Wtime() - mpi_start_time;

        mpi_start_time = MPI_Wtime();
        MPI_Waitall(2, requests, waits);
        acc_times[3*i+2] += MPI_Wtime() - mpi_start_time;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(acc_times, global_acc_times, types * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    for (int i = 0; i < types * 3; i++) global_acc_times[i] /= numprocs;

    if (myproc == 0) {
      std::ofstream myfile;
      myfile.open("pingpong.csv");
      myfile << "Count,MPI_Irecv,MPI_Isend,MPI_Waitall\n";

      for (int i = 0; i < types; i++) {
        myfile << double_counts[i] << "," << global_acc_times[3*i] * 1000 << "," <<
          global_acc_times[3*i+1] * 1000 << "," << global_acc_times[3*i+2] * 1000 << "\n";
      }

      myfile.close();
    }
  }
#endif

#if 1
  /* Internal halo exchange test */
  static bool start = false;

  if (!start) {
    start = true;
    std::vector<std::vector<int>> countss {
      {40000},
      {200, 40000, 40000},
      {1, 200, 200, 200, 40000, 40000, 40000},
      {1, 1, 200, 200, 200, 200, 200, 40000, 40000, 40000, 40000},
      {1, 1, 1, 1, 200, 200, 200, 200, 200, 200, 200, 200, 40000, 40000, 40000, 40000, 40000}
    };
    MPI_Request reqs[40];
    MPI_Status stats[40];
    double* recv_buf = (double*)std::malloc(sizeof(double) * 1048576);
    double* send_buf = (double*)std::malloc(sizeof(double) * 1048576);
    double acc_times[3] = {0.0};
    double global_acc_times[3] = {0.0};

    std::ofstream myfile;
    if (myproc == 0) {
      myfile.open("halo.csv");
      myfile << "Neighbors,MPI_Irecv,MPI_Isend,MPI_Waitall\n";
    }

    for (auto& counts : countss) {
      MPI_Barrier(MPI_COMM_WORLD);
      int num_neighbors = counts.size();

      for (int i = 0; i < 1000; i++) {
        mpi_start_time = MPI_Wtime();
        for (int j = 0; j < num_neighbors; j++) {
          int count = counts[j];
          MPI_Irecv(recv_buf, count, MPI_DOUBLE, 1-myproc, MPI_MY_TAG, MPI_COMM_WORLD, &reqs[j]);
        }
        acc_times[0] += MPI_Wtime() - mpi_start_time;

        mpi_start_time = MPI_Wtime();
        for (int j = 0; j < num_neighbors; j++) {
          int count = counts[j];
          MPI_Isend(send_buf, count, MPI_DOUBLE, 1-myproc, MPI_MY_TAG, MPI_COMM_WORLD, &reqs[num_neighbors+j]);
        }
        acc_times[1] += MPI_Wtime() - mpi_start_time;

        mpi_start_time = MPI_Wtime();
        MPI_Waitall(2*num_neighbors, reqs, stats);
        acc_times[2] += MPI_Wtime() - mpi_start_time;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Reduce(acc_times, global_acc_times, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      for (int i = 0; i < 3; i++) global_acc_times[i] /= numprocs;
      if (myproc == 0) {
        myfile << num_neighbors << "," << global_acc_times[0] * 1000 << "," << global_acc_times[1] * 1000 << "," << global_acc_times[2] * 1000 << "\n";
      }
    }

    if (myproc == 0) {
      myfile.close();
    }
  }
#endif

  MPI_Request mpi_request[num_neighbors*2];
  MPI_Status mpi_status[num_neighbors*2];

  MPI_Barrier(MPI_COMM_WORLD);
  int neighbor_types[num_neighbors] = {0};

  // Post receives first
  for(int i=0; i<num_neighbors; ++i) {
    int n_recv = recv_length[i];
    mpi_start_time = MPI_Wtime();
    MPI_Irecv(x_external, n_recv, mpi_dtype, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, &mpi_request[i]);
    x_external += n_recv;

    int type;
    if (n_recv == 1) type = 0;
    if (n_recv > 100 && n_recv < 300) type = 1; // Count 200
    if (n_recv > 39000 && n_recv < 41000) type = 2; // Count 40000
    neighbor_types[i] = type; // For MPI_Wait
    if (call_counts != NULL) call_counts[type]++; // Only at receive

    if (times != NULL) times[1+type*3] = MPI_Wtime() - mpi_start_time;
  }

#ifdef MINIFE_DEBUG
  os << "launched recvs\n";
#endif


  //
  // Send to each neighbor
  //

#ifdef GPUDIRECT
  Scalar* s_buffer = thrust::raw_pointer_cast(&A.d_send_buffer[0]);
#else
  Scalar* s_buffer = &send_buffer[0];
#endif
  //wait for packing or copy to host to finish
  //cudaEventSynchronize(CudaManager::e1);
  //cudaCheckError();

  for(int i=0; i<num_neighbors; ++i) {
    int n_send = send_length[i];
    mpi_start_time = MPI_Wtime();
    MPI_Isend(s_buffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD, &mpi_request[num_neighbors+i]);
    s_buffer += n_send;

    int type;
    if (n_send == 1) type = 0;
    if (n_send > 100 && n_send < 300) type = 1; // Count 200
    if (n_send > 39000 && n_send < 41000) type = 2; // Count 40000

    if (times != NULL) times[2+type*3] = MPI_Wtime() - mpi_start_time;
  }

#ifdef MINIFE_DEBUG
  os << "send to " << num_neighbors << std::endl;
#endif

  //
  // Complete the reads issued above
  //

  /*
  MPI_Status status;
  for(int i=0; i<num_neighbors; ++i) {
    mpi_start_time = MPI_Wtime();
    if (MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    int type = neighbor_types[i];
    if (times != NULL) times[3+type*3] = MPI_Wtime() - mpi_start_time;
  }
  */
  mpi_start_time = MPI_Wtime();
  MPI_Waitall(num_neighbors*2, mpi_request, mpi_status);
  if (times != NULL) {
    times[3] = MPI_Wtime() - mpi_start_time;
    times[6] = times[9] = 0.0;
  }
  
#ifndef GPUDIRECT
  x.copyToDeviceAsync(local_nrow,CudaManager::s1);
#endif

#ifdef MINIFE_DEBUG
  os << "leaving exchange_externals"<<std::endl;
#endif

//endif HAVE_MPI
#endif
}

#ifdef HAVE_MPI
static std::vector<MPI_Request> exch_ext_requests;
#endif

template<typename MatrixType,
         typename VectorType>
void
begin_exchange_externals(MatrixType& A,
                         VectorType& x)
{
#ifdef HAVE_MPI

  int numprocs = 1, myproc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  if (numprocs < 2) return;

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  // Extract Matrix pieces

  int local_nrow = A.rows.size();
  int num_neighbors = A.neighbors.size();
  const std::vector<LocalOrdinal>& recv_length = A.recv_length;
  const std::vector<int>& neighbors = A.neighbors;

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  exch_ext_requests.resize(num_neighbors*2);

  //
  // Externals are at end of locals
  //
#ifdef GPUDIRECT
  Scalar * x_external = thrust::raw_pointer_cast(&x.d_coefs[local_nrow]);
#else
  std::vector<Scalar>& x_coefs = x.coefs;
  Scalar* x_external = &(x_coefs[local_nrow]);
#endif

  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

  // Post receives first
  for(int i=0; i<num_neighbors; ++i) {
    int n_recv = recv_length[i];
    MPI_Irecv(x_external, n_recv, mpi_dtype, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, &exch_ext_requests[i]);
    x_external += n_recv;
  }

  //
  // Fill up send buffer
  //
  int BLOCK_SIZE=256;
  int BLOCKS=min((int)(A.d_elements_to_send.size()+BLOCK_SIZE-1)/BLOCK_SIZE,2048);

  cudaEventRecord(CudaManager::e1,CudaManager::s1);
  cudaStreamWaitEvent(CudaManager::s2,CudaManager::e1,0);

  copyElementsToBuffer<<<BLOCKS,BLOCK_SIZE,0,CudaManager::s2>>>(thrust::raw_pointer_cast(&x.d_coefs[0]),
                                     thrust::raw_pointer_cast(&A.d_send_buffer[0]), 
                                     thrust::raw_pointer_cast(&A.d_elements_to_send[0]),
                                     A.d_elements_to_send.size());
  cudaCheckError();
  //This isn't necessary for correctness but I want to make sure this starts before the interrior kernel
  cudaStreamWaitEvent(CudaManager::s1,CudaManager::e2,0); 
#ifndef GPUDIRECT
  std::vector<Scalar>& send_buffer = A.send_buffer;
  cudaMemcpyAsync(&send_buffer[0],thrust::raw_pointer_cast(&A.d_send_buffer[0]),sizeof(Scalar)*A.d_elements_to_send.size(),cudaMemcpyDeviceToHost,CudaManager::s2);
  cudaCheckError();
#endif
  cudaEventRecord(CudaManager::e2,CudaManager::s2);
#endif
}

template<typename MatrixType,
         typename VectorType>
inline
void
finish_exchange_externals(MatrixType &A, VectorType &x)
{
#ifdef HAVE_MPI
  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
  
  const std::vector<LocalOrdinal>& send_length = A.send_length;
  const std::vector<int>& neighbors = A.neighbors;
  int num_neighbors = A.neighbors.size();
  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();
  int MPI_MY_TAG = 99;
  
  //
  // Send to each neighbor
  //

#ifdef GPUDIRECT
  Scalar* s_buffer = thrust::raw_pointer_cast(&A.d_send_buffer[0]);
#else
  Scalar* s_buffer = &A.send_buffer[0];
#endif
  
  //wait for packing or copy to host to finish
  cudaEventSynchronize(CudaManager::e2);
  cudaCheckError();

  for(int i=0; i<num_neighbors; ++i) {
    int n_send = send_length[i];
    MPI_Isend(s_buffer, n_send, mpi_dtype, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD, &exch_ext_requests[num_neighbors+i]);
    s_buffer += n_send;
  }
  //
  // Complete the reads issued above
  //

  MPI_Status status;
  for(int i=0; i<exch_ext_requests.size(); ++i) {
    if (MPI_Wait(&exch_ext_requests[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

//endif HAVE_MPI
#endif
}

}//namespace miniFE

#endif

