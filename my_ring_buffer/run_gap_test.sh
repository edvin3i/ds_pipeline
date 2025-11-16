#!/bin/bash
cd /home/nvidia/deep_cv_football/my_ring_buffer
timeout 10 python3 test_tee_ringbuffer.py 2>&1 | tee /tmp/gap_final.log
