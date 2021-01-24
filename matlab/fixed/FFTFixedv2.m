function OutputSignal = FFTFixedv2(InputSignal,bits)
%FFTFIXED Fixed-Point fft transform
% Input Signal is the input
% Increase the compute speed
bit_twiddle = bits+2;
S = InputSignal;
FFTn = length(S(:,1));                                                      % The number of FFT
k = length(S(1,:));                                                         % How many times
N = FFTn;
W=fi(exp(-1*2j*pi*(0:N-1)/N), 1, bit_twiddle, bit_twiddle-1);                                   % fi(v, s, w, f, F)returns a fixed-point object with value v,signed property values
OutputSignal = fi(zeros(FFTn,k),1,bits,bits-1);
p=nextpow2(FFTn);                                                           % checking the size of the input array


temp = fi(S,1,bits,bits-1);
if FFTn~=2^p
    disp('fix the size of signal to FFT');
    temp = [temp; zeros(2^p-length(temp(:,1)),length(temp(1,:)))];
end

s = bitrevorder(temp);
Stages = log2(N);
%----------------------------------DIT------------------------------------- 
% shift and roundoff at each stage 
    Half=1;
    for stage = 1:Stages
        for index=0:(N/(2^(Stages-stage))):(N-1)
            for n=0:Half-1
 
                pos=n+index+1;                                             % index of the data sample
                pow=(2^(Stages-stage))*n;                                  % part of power of the complex multiplier
                w = W(pow+1);
%                 a=bitsra(s(pos)+fi(s(pos+Half).*w,1,bits,bits-1),1);             % 1-st part of the "butterfly" creating operation shift & roundoff
%                 b=bitsra(s(pos)-fi(s(pos+Half).*w,1,bits,bits-1),1);             % 2-nd part of the "butterfly" creating operation
                
                a=bitsra(s(pos,:)+fi((fi(real(s(pos+Half,:)).*real(w),1,bits,bits-1)-fi(imag(s(pos+Half,:)).*imag(w),1,bits,bits-1))+...
                    (fi(imag(s(pos+Half,:)).*real(w),1,bits,bits-1)+fi(real(s(pos+Half,:)).*imag(w),1,bits,bits-1))*1j,1,bits,bits-1),1); 
                b=bitsra(s(pos,:)-fi((fi(real(s(pos+Half,:)).*real(w),1,bits,bits-1)-fi(imag(s(pos+Half,:)).*imag(w),1,bits,bits-1))+...
                    (fi(imag(s(pos+Half,:)).*real(w),1,bits,bits-1)+fi(real(s(pos+Half,:)).*imag(w),1,bits,bits-1))*1j,1,bits,bits-1),1);
                
                
                
                s(pos,:)=a;                                                  % saving computation of the 1-st part
                s(pos+Half,:)=b;
            end
        end
        Half=Half*2;
    end

    OutputSignal = s;
end




% 
% 
% for i = 1:k
%     s = fi(bitrevorder(S(:,i)),1,bits,bits-1);
%     s = [s zeros(1,(2^p)-length(s))];                                             % complementing an array of zeros if necessary
%     Stages = log2(N);
%     
% %----------------------------------DIF-------------------------------------    
% %     Half=N/2;
% %     for stage = 1:Stages
% %         for index=0:(N/(2^(stage-1))):(N-1)
% %             for n=0:Half-1
% %  
% %                 pos=n+index+1;                                                  % index of the data sample
% %                 pow=(2^(stage-1))*n;                                            % part of power of the complex multiplier
% %                 w = W(pow+1);
% %                 a=s(pos)+s(pos+Half);                                           % 1-st part of the "butterfly" creating operation
% %                 b=(s(pos)-s(pos+Half)).*w;                                      % 2-nd part of the "butterfly" creating operation
% %                 
% %                 s(pos)=a;                                                       % saving computation of the 1-st part
% %                 s(pos+Half)=b;
% %             end
% %         end
% %         Half=Half/2;
% %     end
% 
% %----------------------------------DIT------------------------------------- 
% % shift and roundoff at each stage 
%     Half=1;
%     for stage = 1:Stages
%         for index=0:(N/(2^(Stages-stage))):(N-1)
%             for n=0:Half-1
%  
%                 pos=n+index+1;                                             % index of the data sample
%                 pow=(2^(Stages-stage))*n;                                  % part of power of the complex multiplier
%                 w = W(pow+1);
% %                 a=bitsra(s(pos)+fi(s(pos+Half).*w,1,bits,bits-1),1);             % 1-st part of the "butterfly" creating operation shift & roundoff
% %                 b=bitsra(s(pos)-fi(s(pos+Half).*w,1,bits,bits-1),1);             % 2-nd part of the "butterfly" creating operation
%                 
%                 a=bitsra(s(pos)+fi((fi(real(s(pos+Half)).*real(w),1,bits,bits-1)-fi(imag(s(pos+Half)).*imag(w),1,bits,bits-1))+...
%                     (fi(imag(s(pos+Half)).*real(w),1,bits,bits-1)+fi(real(s(pos+Half)).*imag(w),1,bits,bits-1))*1j,1,bits,bits-1),1); 
%                 b=bitsra(s(pos)-fi((fi(real(s(pos+Half)).*real(w),1,bits,bits-1)-fi(imag(s(pos+Half)).*imag(w),1,bits,bits-1))+...
%                     (fi(imag(s(pos+Half)).*real(w),1,bits,bits-1)+fi(real(s(pos+Half)).*imag(w),1,bits,bits-1))*1j,1,bits,bits-1),1);
%                 
%                 
%                 
%                 s(pos)=a;                                                  % saving computation of the 1-st part
%                 s(pos+Half)=b;
%             end
%         end
%         Half=Half*2;
%     end
% 
%     OutputSignal(:,i) = s;
% end
% 
% 
% end
