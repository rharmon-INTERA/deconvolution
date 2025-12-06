clear all;
clear all;

prefix = 'gamma';
mkdir(prefix)
ins_pth = [prefix,'/inputs/'];
mkdir(ins_pth)
outs_pth = [prefix,'/outputs/'];
mkdir(outs_pth)

% Parameters
N = 512;               % Number of points
dt = 0.1;              % Time step
t = dt * (0:N-1)';     % Time vector


out = zeros(N,1); 
in  = out; 
kernel = out; 
pad = out;

kShape  = 3;   % "shape" parameter 
theta   = 2;   % "scale" parameter

for i = 1:N
    if t(i) >= 0
        kernel(i) = (1 / (gamma(kShape) * theta^kShape)) * ...
                     (t(i)^(kShape - 1)) * exp(-t(i) / theta);
    else
        kernel(i) = 0;
    end
end

% normalize:
kernel = kernel / (dt * sum(kernel));

% Same gauss-based input as chapeau and bimodal:
in = exp(((t - (dt*N/5)).^2) / (-4)) + 0.25 * exp(((t - (dt*N/3)).^2) / (-8));
in = 10 * in;  % Scale up the input by a factor of 10

out = dt * conv([in; pad], [kernel; pad], 'full');
out = out(1:N);

figure;
subplot(2,1,1)
plot(t, kernel, 'LineWidth', 1.5);
title('Gamma Function Kernel');
xlabel('Time');
ylabel('Amplitude');

subplot(2,1,2)
plot(t, in, 'b', t, out, 'r', 'LineWidth', 1.5);
legend('Input Signal','Output (Convolved)');
title('Input Signal and Output');
xlabel('Time');
ylabel('Amplitude');

saveas(gcf, [ins_pth,[prefix,'_in_tfx_out.png']]);
save([ins_pth, [prefix,'.mat']], 'in', 'out', 'kernel', 't');
T = table(t, in, out, kernel, 'VariableNames', {'Time', 'Input', 'Output', 'Kernel'});
writetable(T, [ins_pth,[prefix,'.csv']]);