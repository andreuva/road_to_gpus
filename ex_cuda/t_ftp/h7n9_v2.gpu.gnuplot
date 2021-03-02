#
# usage:   gnuplot  "./h7n9_v2.gpu.gnuplot"                   
#
# purpose: just visualize results obtained from program h7n9_v2.cu
#
 set terminal postscript eps enhanced solid  color
 set encoding iso_8859_1
 set output "h7n9_v2.gpu.eps"
 set size square
 set ylabel "H7N9 Viral Load [\#particles]" offset  -0.5, 0 font "Bold, 20"   
 set xlabel "Time Post Infection [h]" offset 0, -0.5 font "Bold, 20"
 set nokey 
 plot "h7n9_v2.gpu.1.dat" using 1:8 with lines lw 3


