      PROGRAM FVCADD
C
C     PURPOSE:  COMBINE PLAIN FORTRAN CODE WITH MINIMAL CUDA ROUTINE INCLUDING
C               A KERNEL CALL TO THE GPU FOR THE ACTUAL COMPUTATION
C
C     COMPILE:  gfortran -c fvcadd.f
C               nvcc -c ntmdtr.cu     
C               nvcc fvcadd.o ntmdtr.o -lcudart -lgfortran
C

      IMPLICIT NONE
      INTEGER N
      PARAMETER (N = 100)
      INTEGER I
      REAL A(N), B(N), C(N)

      DO I=1, N
         A(I) = REAL(I)
         B(I) = REAL(N) - REAL(I)
         C(I) = REAL(0) 
      ENDDO

C
C     CALL CUDA PART FROM WITHIN AN INTERMEDIARY MEDIATOR CODE 
C
      CALL NTMDTR(A, B, C, N)

      DO I=1, N
         WRITE(6, '(I6F12.6)') I, C(I)
      ENDDO 

      END
