x = load('data/part23.mat'); 
x = x.mydata; 
fs = 1000; df = 5; 
dalpha = 0.5;

[Sxx,alphao,fo]=autofam(x,fs,df,dalpha);

Sxx_max = (max(Sxx))';

X = load('data/part23_alpha.mat'); 
X = X.mydata; 
Y = fliplr(X); 
Z = [0, X, Y(2:end)]';

RMSE = sqrt(mean((Sxx_max-Z).^2))
NRMSE = sqrt(mean((Sxx_max-Z).^2)) / (max(Sxx_max)-min(Sxx_max))

pererr = mean(abs(Sxx_max-Z)./Sxx_max*100);