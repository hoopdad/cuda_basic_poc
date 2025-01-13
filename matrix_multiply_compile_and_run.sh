#!/bin/bash
nvcc matrix_multiply.cu -o matrix_multiply.out && \
  ./matrix_multiply.out
