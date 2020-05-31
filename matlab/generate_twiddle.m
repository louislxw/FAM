% Program for generating n-length FFT's twiddle factor
% By: Denny Hermawanto
% Puslit Metrologi LIPI, INDONESIA
% Copyright 2015
clear all;
close all;
fft_length = input('Enter FFT length:');
for mm = 0:1:(fft_length-1)
    theta = (-2*pi*mm*1/fft_length);
%   Twiddle factor equation
%   twiddle = exp(1i*theta);
%   Euler equation for complex exponential 
%   e^(j*theta) = cos(theta) + j(sin(theta)) 
    
    twiddle(mm+1) = cos(theta) + (1i*(sin(theta)));
    real_twiddle = real(twiddle);
    real_twiddle = real_twiddle';
    im_twiddle = imag(twiddle);
    im_twiddle = im_twiddle';
    
end;