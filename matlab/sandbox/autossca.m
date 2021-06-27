function [Sx,alphao,fo]=autossca(x,fs,df,dalpha)
%   AUTOSSCA(X,FS,DF,DALPHA) computes the spectral auto-
%   correlation density function estimate of the signals X,
%   by using the Strip Spectral Correlation Algorithm (SSCA). 
%   Make sure that DF is much bigger than DALPHA in order to 
%   have a reliable estimate.
%
%   INPUTS:
%   X       - input column vector;
%   FS      - sampling rate;
%   DF      - desired frequency resolution;
%   DALPHA  - desired cyclic frequency resolution.
%
%   OUTPUTS:
%   SX      - spectral correlation density function estimate;
%   ALPHAO  - cyclic frequency; and
%   FO      - spectrum frequency.
%
%   Credits to E.L.Da Costa
%
% e.g. x = (1:2240)'; fs = 1000; df = 5; dalpha = 0.5; % Add by Louis
% or x = repmat((0:0.1:0.3)', 560, 1); fs = 1000; df = 5; dalpha = 0.5; % Add by Louis

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
if length(x) < N
    x(N)=0;
    disp('you will not get the desired resolution in cyclic frequency');
    dalpha=fs/N;
    disp(['cyclic frequency resolution=', num2str(dalpha)]);
elseif length(x) > N
    x=x(1:N);
end

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

%%%%%%%%%%%%%%%
%% First FFT %%
%%%%%%%%%%%%%%%
XF1=fft(XW);
XF1=fftshift(XF1, 1);
XF1=[XF1(:, P/2+1:P) XF1(:, 1:P/2)];

%%%%%%%%%%%%%%%%%%%%
%% Downconversion %%
%%%%%%%%%%%%%%%%%%%%
E=zeros(Np,P);

for k=-Np/2:Np/2-1
    for m=0:P-1
        E(k+Np/2+1,m+1)=exp(-i*2*pi*k*m*L/Np);
    end
end

XD=XF1.*E;

%%%%%%%%%%%%%%%%%
%% Replication %%
%%%%%%%%%%%%%%%%%
XR=zeros(Np,P*L);

for k=1:P
    XR(:,(k-1)*L+1:k*L) = XD(:,k)*ones(1,L);
end

%%%%%%%%%%%%%%%%%%%%
%% Multiplication %%
%%%%%%%%%%%%%%%%%%%%
xc=ones(Np,1)*x';
XM=XR.*xc;
XM=conj(XM');

%%%%%%%%%%%%%%%%
%% Second FFT %%
%%%%%%%%%%%%%%%%
XF2=fft(XM);
XF2=fftshift(XF2);
XF2=[XF2(:,Np/2+1:Np) XF2(:,1:Np/2)];
M=abs(XF2);
alphao=(-1:1/N:1); % *fs ?
fo=(-.5:1/Np:.5); % *fs ?
Sx=zeros(Np+1,2*N+1);

%%%%%%%%%%%%%%%%%%%
%% alpha profile %%
%%%%%%%%%%%%%%%%%%%
for k1=1:N
    for k2=1:Np
        alpha=(k1-1)/N+(k2-1)/Np-1;
        f=((k2-1)/Np-(k1-1)/N)/2;
        k=ceil(1+Np*(f+.5));
        l=1+N*(alpha+1);
        Sx(k,l)=M(k1,k2);
    end
end

