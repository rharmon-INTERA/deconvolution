clear all

prefix='chapeau';
mkdir(prefix)
ins_pth = [prefix,'/inputs/'];
mkdir(ins_pth)
outs_pth = [prefix,'/outputs/'];
mkdir(outs_pth)

N=512;        % Number of points in input, kernel, and output functions
dt=0.1;       % constant delta t
L=dt*(N-1);   % Time length of signals

out=zeros(N,1); in=out; kernel=out; pad=out;

peak=floor(N/20);

kernel(1:peak)=0:peak-1;
kernel(peak+1:peak+peak+1)=peak:-1:0;
kernel=kernel/(dt*sum(kernel));
kernel=kernel;

t=dt*(0:N-1)';
in=exp(( (t - (dt*N/5)) .^2)/(-4)) + 0.25* exp(( (t - (dt*N/3)) .^2)/(-8));
in=10*in;

out=dt*conv([in ; pad], [kernel ; pad],'full');
out=out(1:N);
%out=ifft(fft(kernel).*fft(in));
%out=dt*out(1:N);
figure(111)
plot(t,in(1:N),t,out(1:N),t,kernel(1:N))
legend('in','out','kernel')

figure;
subplot(2,1,1)
plot(t, kernel, 'LineWidth', 1.5);
title('Chapeau Transfer Function');
xlabel('Time');
ylabel('Amplitude');

subplot(2,1,2)
plot(t, in, 'b', t, out, 'r', 'LineWidth', 1.5);
legend('Input Signal','Output (Convolved with transfer function)');
title('Input Signal and Output');
xlabel('Time');
ylabel('Amplitude');

m_0=dt*sum(kernel)
m_1=(dt/m_0)*sum(t.*kernel)
var_exp=(dt/m_0)*sum(((t-m_1).^2).*kernel) 

saveas(gcf, [ins_pth,[prefix,'_in_tfx_out.png']]);
save([ins_pth, [prefix,'.mat']], 'in', 'out', 'kernel', 't');
T = table(t, in, out, kernel, 'VariableNames', {'Time', 'Input', 'Output', 'Kernel'});
writetable(T, [ins_pth,[prefix,'.csv']]);
