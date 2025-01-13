#!/bin/bash
nvcc vector_add.cu -o vector_add.out && \
  ./vector_add.out
