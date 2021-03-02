#
# usage:   gnuplot  "./h7n9_v4.gnuplot"                         
# purpose: fit the obtained immune-response-modulated growth pattern
#          of some h7n9 virus currently stored to file './h7n9_v4.1.dat'
#          with a gaussian function;
#
 set terminal postscript eps enhanced solid  color
 set encoding iso_8859_1
 set output "h7n9_v4.eps"
 set ylabel "H7N9 Viral Load [\#particles]" offset  -0.3, 0 font "Bold, 20"   
 set xlabel "Time Post Infection [h]" offset 0, -0.3 font "Bold, 20"
 set nokey 
 set xrange [0:500]
 a=1.1e+09
 b=-0.005
 c=139.9
 immnty(x)=a*exp(b*(x-c)*(x-c))
 fit immnty(x) "./h7n9_v4.1.dat" using 1:2 via a,b,c
 set label 1 "g(x) = a*exp(b*(x-c)*(x-c))" at 300.0, 1.1e+09 font "Times-Bold, 20" tc lt 7
 set label 2 "a = 3.78678e+08"             at 300.0, 1.0e+09 font "Times-Bold, 20" tc lt 7
 set label 3 "b = -0.0015469"              at 300.0, 9.0e+08 font "Times-Bold, 20" tc lt 7
 set label 4 "c = 140.801    "             at 300.0, 8.0e+08 font "Times-Bold, 20" tc lt 7
 plot "./h7n9_v4.1.dat" using 1:2 with lines lw 3, immnty(x) with linespoints lw 3


