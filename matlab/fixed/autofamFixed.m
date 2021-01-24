function [Sx,alphao,fo,result]=autofamFixed(x,fs,df,dalpha,bitInput)
%   AUTOFAM(X,FS,DF,DALPHA) computes the spectral auto-
%   correlation density function estimate of the signals X
%   by using the FFT Accumulation Method(FAM). Make sure that
%   DF is much bigger than DALPHA in order to have a reliable estimate.
%
%   INPUTS:
%   X       - input column vector;
%   FS      - sampling rate;
%   DF      - desired frequency resolution;
%   DALPHA  - desired cyclic frequency resolution
%
%   OUTPUTS:
%   SX     - spectral correlation density function estimate;
%   ALPHAO  - cyclic frequency;
%   FO      - spectrum frequency;
%
% e.g. x = (1:2240)'; fs = 1000; df = 5; dalpha = 0.5; % Add by Louis
% or x = repmat(0:0.1:0.3, [1 560]); fs = 1000; df = 5; dalpha = 0.5; % Add by Louis

if nargin ~= 5
    error('Wrong number of arguments.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Definition of Parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Np=pow2(nextpow2(fs/df));       %Number of input channels, defined
                                %by the desired frequency
                                %resolution (df) as follows:
                                %Np=fs/df, where fs is the original
                                %data sampling rate. It must be a
                                %power of 2 to avoid truncation or
                                %zero-padding in the FFT routines;
                                
L=Np/4;                         %Offset between points in the same
                                %column at consecutive rows in the
                                %same channelization matrix. It
                                %should be chosen to be less than
                                %or equal to Np/4;
                                
P=pow2(nextpow2(fs/dalpha/L));  %Number of rows formed in the
                                %channelization matrix, defined by
                                %the desired cyclic frequency
                                %resolution (dalpha) as follows:
                                %P=fs/dalpha/L. It must be a power
                                %of 2;
                                
N=P*L;                          %Total number of points in the
                                %input data

                                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% bits setting for each block %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bit.input = 16;
windowing = 16;
% bit.firstFFT = 16;
% bit.ConjMulti = 17;
% bit.secondFFT = 18;
bit = bitInput;
                                
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input Channelization %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%ylb: this is zero padding if necessary and taking truncating if not.
if length(x) < N
    x(N)=0;
elseif length(x) > N
    x=x(1:N);
end
% n = sqrt(2^-30/12)*randn(size(x));
% xt = x;
% x = xt+n;
% snr(xt,xt-x)
NN=(P-1)*L+Np;
xx=x;
xx(NN)=0;
xx=xx(:);
X=fi(zeros(Np,P),1,bit,bit-1); %%
for k=0:P-1
    X(:,k+1)=xx(k*L+1:k*L+Np);
end

% n = sqrt(2^-30/12)*randn(size(X));
% xt = X;
% X = xt+n;
% 
% disp('SQNR of Framing = ')
% snr(xt,xt-X)
result.Input = X;

%%%%%%%%%%%%%%%
%% Windowing %%
%%%%%%%%%%%%%%%
a=hamming(Np);
XW=fi(diag(a)*X,1,windowing,windowing-1);
result.Windowing = XW;
% xw = diag(a)*xt;
% n = sqrt(2^-30/12)*randn(size(XW));
% XW = XW + n;
% disp('SQNR of Windowing = ')
% snr(XW,n)
% disp('SQNR of Previous Steps = ')
% snr(xw,xw-XW)
% XW=int16(XW); % Add by Louis

%%%%%%%%%%%%%%%
%% First FFT %%
%%%%%%%%%%%%%%%
% XF1=FFTFixed(XW,bit.firstFFT);
XF1=FFTFixedv2(XW,bit.firstFFT);
refXF1 = FFTFloat(single(XW));


%These two statements perform the same operation
%as simply doing the new third operation
%XF1=fftshift(XF1);
%XF1=[XF1(:,P/2+1:P) XF1(:,1:P/2)];
XF1=fftshift(XF1, 1);
result.FirstFFT = XF1;
%surf(log(abs(XF1)));
%title('FFT of data');

%%%%%%%%%%%%%%%%%%%%
%% Downconversion %%
%%%%%%%%%%%%%%%%%%%%

% This is doing a time shift to account for the shift in L on the data prior.  each column gets a shift of column # * L.

E=zeros(Np,P);
for k=-Np/2:Np/2-1
    for m=0:P-1
        E(k+Np/2+1,m+1)=exp(-i*2*pi*k*m*L/Np);
    end
end

XD=XF1.*E;

%all this operation does is take a transpose and 2 complex conjugates
%so the complex conjugate operations cancel each other out
%XD=conj(XD');
XD=transpose(XD);  % Prior to this, each column was a set of time, column 1 was t = 0 ... 4, column 2 was t = 2 .... 6 etc.
% Now each row is time t = 0 .... 4, row 2 t = 2 .... 6 etc.

%figure;surf(log(abs(XD))); title('Down conversion Trans');

%%%%%%%%%%%%%%%%%%%%
%% Multiplication %%
%%%%%%%%%%%%%%%%%%%%

%ylb so what the multiplication is doing, it puts a copy of the FFT in, scaled by FFT(0) then again by FFT(1) etc for all time.  
% It's similar to the auto correlation of the FFT.
XM=fi(zeros(P,Np^2),1,bit.ConjMulti,bit.ConjMulti-1);

scale = 2^0;

maxvalue = max(max(max(abs(real(XD)))),max(max(abs(imag(XD)))));
if maxvalue<=1/2
    scal = 2^(ceil(log2(1/2/single(maxvalue))));
end

XD = fi(XD*scal,1,bit.ConjMulti,bit.ConjMulti-1);
scale = scale * scal;


for k=1:Np
    for l=1:Np
%         XM(:,(k-1)*Np+l)=fi((XD(:,k).*conj(XD(:,l))),1,bit.ConjMulti,bit.ConjMulti-1);
        
        XM(:,(k-1)*Np+l) = (fi(real(XD(:,k)).*real(conj(XD(:,l))),1,bit.ConjMulti,bit.ConjMulti-1)...
            -fi(imag(XD(:,k)).*imag(conj(XD(:,l))),1,bit.ConjMulti,bit.ConjMulti-1))...
            +(fi(imag(XD(:,k)).*real(conj(XD(:,l))) ,1,bit.ConjMulti,bit.ConjMulti-1)...
            +fi( real(XD(:,k)).*imag(conj(XD(:,l))),1,bit.ConjMulti,bit.ConjMulti-1))*1j;
    end
end
%rescale
%XM.*2^(-1*fix(log2(double(max(max([real(XM),imag(XM)]))))));
%-----------------
result.Multi = XM;
result.Scale.Multi = scale;
%figure;surf(log(abs(XM))); title('Post Multiplication');

%%%%%%%%%%%%%%%%
%% Second FFT %%
%%%%%%%%%%%%%%%%
% XF2=FFTFixed(XM,bit.secondFFT);
XF2=FFTFixedv2(XM,bit.secondFFT);
%XF2=fftshift(XF2);
%XF2=[XF2(:,Np^2/2+1:Np^2) XF2(:,1:Np^2/2)];
XF2=fftshift(XF2,1);
result.SecondFFT = XF2;
%figure;surf(log(abs(XF2))); title('FFT 2');

% Here he is cutting out the high frequency and low frequency components, why?  
XF2=XF2(P/4+1:3*P/4, :); % XF2=XF2(P/4:3*P/4, :); % Changed by Louis

%Get the magnitude
M=fi(abs(XF2),1,16,15);
alphao=-1:1/N:1;
fo=-.5:1/Np:.5;
Sx=zeros(Np+1,2*N+1);

% The size of M  is (P/2, Np^2)
% The size of Sx is (Np+1, 2*N+1)

%%%%%%%%%%%%%%%%%%%
%% alpha profile %%
%%%%%%%%%%%%%%%%%%%
for k1=1:P/2 % k1=1:P/2+1 % Changed by Louis
    for k2=1:Np^2
        if rem(k2,Np)==0
            l=Np/2-1;
        else
            l=rem(k2,Np)-Np/2-1;
        end
	      
        k=ceil(k2/Np)-Np/2-1;
        p=k1-P/4-1;
        alpha=(k-l)/Np+(p-1)/L/P;
        f=(k+l)/2/Np;
        
        if alpha<-1 || alpha>1
            k2=k2+1;
        elseif f<-.5 || f>.5
            k2=k2+1;
        else
            kk=ceil(1+Np*(f+.5));
            ll=1+N*(alpha+1);
            Sx(kk,ll)=M(k1,k2);
        end
    end
end
result.Sx = Sx;
end
