function [Sx,Sx_fi,Sx_int,alphao,fo]=autofam_fixed(x,fs,df,dalpha)
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
% e.g. x = (1:2240)'; fs = 1000; df = 5; dalpha = 0.5; 
% or x = repmat(0:0.1:0.3, [1 560]); fs = 1000; df = 5; dalpha = 0.5; 
% or x = load('iq.mat'); x = x.mydata; fs = 1000; df = 5; dalpha = 0.5; 

if nargin ~= 4
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

%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input Channelization %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%ylb: this is zero padding if necessary and taking truncating if not.
% if length(x) < N
%     x(N)=0;
% elseif length(x) > N
%     x=x(1:N);
% end

NN=(P-1)*L+Np;
xx=x;
xx(NN)=0;
xx=xx(:);
X=zeros(Np,P);
for k=0:P-1
    X(:,k+1)=xx(k*L+1:k*L+Np);
end

%%%%%%%%%%%%%%%
%% Windowing %%
%%%%%%%%%%%%%%%
a=hamming(Np);
XW=diag(a)*X;
% XW=int16(XW); % Add by Louis

%%%%%%%%%%%%%%%
%% First FFT %%
%%%%%%%%%%%%%%%
XF1=fft(XW);
%These two statements perform the same operation
%as simply doing the new third operation
%XF1=fftshift(XF1);
%XF1=[XF1(:,P/2+1:P) XF1(:,1:P/2)];
XF1=fftshift(XF1, 1);
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

% figure;surf(log(abs(XD))); title('Down conversion Trans');
XD_fi=fi(XD,1,16,10); % 6.10
XD_int=int32(XD*(2^10)); % 32-bit integer
XD_single=single(XD_int);
% figure;surf(real(XD_fi)); title('Down conversion real');
% figure;surf(imag(XD_fi)); title('Down conversion image');
%%%%%%%%%%%%%%%%%%%%
%% Multiplication %%
%%%%%%%%%%%%%%%%%%%%

%ylb so what the multiplication is doing, it puts a copy of the FFT in, scaled by FFT(0) then again by FFT(1) etc for all time.  
% It's similar to the auto correlation of the FFT.
XM=zeros(P,Np^2);
XM_fi=fi(zeros(P,Np^2),1,16,4); % 12.4
XM_int=int32(zeros(P,Np^2));
for k=1:Np
    for l=1:Np
        XM(:,(k-1)*Np+l)=(XD(:,k).*conj(XD(:,l)));
        XM_fi(:,(k-1)*Np+l)=(XD_fi(:,k).*conj(XD_fi(:,l)));
        XM_int(:,(k-1)*Np+l)=(XD_single(:,k).*conj(XD_single(:,l)));
%         XM_fi(:,(k-1)*Np+l) ...
%         = (fi(real(XD_fi(:,k)).*real(conj(XD_fi(:,l))),1,16,4)...
%           -fi(imag(XD_fi(:,k)).*imag(conj(XD_fi(:,l))),1,16,4))...
%          +(fi(imag(XD_fi(:,k)).*real(conj(XD_fi(:,l))) ,1,16,4)...
%           +fi(real(XD_fi(:,k)).*imag(conj(XD_fi(:,l))),1,16,4))*1j;
    end
end
XM_r=real(XM_int); % /(2^20);
XM_i=imag(XM_int); % /(2^20);
XM_int = int32(single(XM_r) + single(XM_i)*1j);
% figure;surf(log(abs(XM))); title('Post Multiplication');
% figure;surf(real(XM_fi)); title('Down conversion real');
% figure;surf(imag(XM_fi)); title('Down conversion image');

% T3 = fi_m_radix2fft_scaled_fixed_types(); % Get fixed point data types declared in table
% x = cast(XM,'like',T3.x);
% w0 = fi_radix2twiddles(32);
% w = cast(w0,'like',T3.w);
% y = fi_m_radix2fft_withscaling_typed(x,w,T3);
% XF2 = y;

%%%%%%%%%%%%%%%%
%% Second FFT %%
%%%%%%%%%%%%%%%%
XF2=fft(XM);
XF2_fi=fft(single(XM_fi));
XF2_int=fft(single(XM_int));
%XF2=fftshift(XF2);
%XF2=[XF2(:,Np^2/2+1:Np^2) XF2(:,1:Np^2/2)];
XF2=fftshift(XF2,1); % float
XF2_fi=fi(fftshift(XF2_fi,1),1,16,2); % 14.2
XF2_int=int32(fftshift(XF2_int,1)); % int32
%figure;surf(log(abs(XF2))); title('FFT 2');

% Here he is cutting out the high frequency and low frequency components, why?  
XF2=XF2(P/4+1:3*P/4, :); % XF2=XF2(P/4:3*P/4, :); % Changed by Louis
XF2_fi=XF2_fi(P/4+1:3*P/4, :);
XF2_int=XF2_int(P/4+1:3*P/4, :);
%Get the magnitude
M=abs(XF2);
M_fi=abs(XF2_fi);
M_int=abs(single(XF2_int)); %
alphao=-1:1/N:1;
fo=-.5:1/Np:.5;
Sx=zeros(Np+1,2*N+1);
Sx_fi=fi(zeros(Np+1,2*N+1),1,16,2); % 14.2
Sx_int=int32(zeros(Np+1,2*N+1)); % 
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
            Sx_fi(kk,ll)=M_fi(k1,k2);
            Sx_int(kk,ll)=M_int(k1,k2);
        end
    end
end
