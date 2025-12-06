clear all;

prefix = 'bimodal';
mkdir(prefix)
ins_pth = [prefix,'/inputs/'];
mkdir(ins_pth)
outs_pth = [prefix,'/outputs/'];
mkdir(outs_pth)

% Parameters
N = 512;            % Number of points
dt = 0.1;           % Time step
t = dt * (0:N-1)';  % Time vector

prefix = 'bimodal';
out=zeros(N,1); in=out; kernel=out; pad=out;

% Define Gaussian parameters
mu1 = dt * N / 5;        % Mean of the first Gaussian
sigma1 = 2.75;            % Standard deviation of the first Gaussian
mu2 = dt * 2.25 * N / 6;    % Mean of the second Gaussian
sigma2 = 2.5;            % Standard deviation of the second Gaussian

% Create the bimodal kernel with the second peak scaled by 0.5
kernel = exp(-((t - mu1).^2) / (2 * sigma1^2)) ...
       + 0.4 * exp(-((t - mu2).^2) / (2 * sigma2^2));

% Normalize the kernel
kernel = kernel / (dt * sum(kernel));

t=dt*(0:N-1)';
in=exp(( (t - (dt*N/5)) .^2)/(-4)) + 0.25* exp(( (t - (dt*N/3)) .^2)/(-8));
in=10*in;

out=dt*conv([in ; pad], [kernel ; pad],'full');
out=out(1:N);

% Plot the kernel

figure;
subplot(2,1,1)
plot(t, kernel, 'LineWidth', 1.5);
title('Bimodal Transfer Function');
xlabel('Time');
ylabel('Amplitude');

subplot(2,1,2)
plot(t, in, 'b', t, out, 'r', 'LineWidth', 1.5);
legend('Input Signal','Output (Convolved with transfer function)');
title('Input Signal and Output');
xlabel('Time');
ylabel('Amplitude');

saveas(gcf, [ins_pth,[prefix,'_in_tfx_out.png']]);
save([ins_pth, [prefix,'.mat']], 'in', 'out', 'kernel', 't');
T = table(t, in, out, kernel, 'VariableNames', {'Time', 'Input', 'Output', 'Kernel'});
writetable(T, [ins_pth,[prefix,'.csv']]);