%function deconv_olaf_condsim
clear
%%%%%%%%%%  Some reach details %%%%%%%%%%%%%%%%%%%%%
prefix='gamma';
load([prefix '.mat']);  % three vectors of equal length: time, in, out
dx=20;                 % m downstream of the upstream
%%%%%%%%%%  Some user-defined numerical details %%%%%%%%%%%%%%%%%%%%%
% slope of the linear variogram
theta=1e-4;       %  A first guess applied linearly to corr_time:
corr_time=60;     %  Only applies if less than the total filter g(t) length
% standard deviation of epistemic error (initial guess)
sigma = .0001;
sigma_max= .1;   % DAB places a forced maximum on iterated sigma
% length of transfer-function vector (dt remains the same)
n_g=400;  
% number of realizations
nreal=50;  % Make 50 for nice final plots - takes a while
method='cirpka';
method='linear';
method='learn';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rand('state',sum(100*clock));
% input signal and output signals
x=in; y=out;

% Add an uncorrelated noise to the known input if desired:
realnoise_sigma = 0.09; rand_mult=realnoise_sigma/sqrt(1/12);
y = y + rand_mult*(rand(size(y))-0.5);   % std of noise = realnoise_sigma

% time increment
dt=t(2)-t(1);
corr_time = min(n_g*dt,corr_time);  % DAB
n_corr_time=ceil(corr_time/dt);     % DAB
n_h_buffer=10;
cov=zeros(1,n_g);                   % First guess
cov(1:n_corr_time)=(theta/n_corr_time)*(n_corr_time:-1:1);  % DAB

theta_old=0; rel_cov_change = 999;
tic
%while ( abs( (theta_old-theta)/theta) > 0.001 )

% Make the Jacobian in y = Xg that encodes output = input * filter:
input_fn=dt*x;
input_fn(input_fn<1e-8)=0;
r=dt*zeros(1,n_g);
X=toeplitz(input_fn,r); 

while ( rel_cov_change > 0.1 )  % Still tinkering with convergence norm and crit

% construction of generalized covariance matrix (here time in units of "time")
c=[n_g:-1:1]*dt*theta;      % Olaf's linear with new theta
c(1:length(cov))=cov;       % DAB     actual cov function

figure(444)
plot(dt*((1:n_g)-1),cov,'-')   % keep track of changing cov function
legend('new COV'); hold on

Q=toeplitz(c);
% This traps slightly (?) non-postive definite Q
%[v,d]=eig(Q); d(d<0)=1e-10;
%Q=v*d*v';
%%%%%%%%%%%%%%%%%%%
C=chol(Q);
invC=inv(Q);
% May need to re-build X if n_g changes...
r=dt*zeros(1,n_g);
X=toeplitz(input_fn,r); 

% vector of indices
ii=[1:n_g]';

% re-initialize storage of filters g for this covariance function
g_all  = zeros(n_g,nreal+1);

% loop over all realizations - the first is a best estimate of only y
ireal=0;
disp(' Getting sigma for non-noised output y')

while ireal<nreal+1   % DAB changed to reject non-converged realizations
    % unconditional realization (set to zero for first "best" solution)
    g_uc=zeros(n_g,1);
    me=zeros(size(y));
    if ireal>0   % add error vectors for ensemble runs
        g_uc=C'*randn(n_g,1);
        me = sigma*randn(size(y));  % measurement error
    end
    % initialization of constraints
    hL=[];
    nLagrange=0;
    iter=0;
    while iter<50         % If it doesn't converge in 50, it probably won't
        iter=iter+1;
        % construction of unconstrained matrix
        XbyX=X'*X/sigma^2;
        u=ones(n_g,1);
        up_left_mat=[XbyX+invC,XbyX*u;u'*XbyX,u'*XbyX*u];
        upper_rhs=[X'*(y+me)/sigma^2-XbyX*g_uc; u'*X'*(y+me)/sigma^2-u'*XbyX*g_uc];
        % matrix related to the Lagrange multipliers
        H_u_mat=zeros(nLagrange,n_g+1);
        Lrhs=zeros(nLagrange,1);
        for j=1:nLagrange
            H_u_mat(j,hL(j))=1;
            H_u_mat(j,n_g+1)=1;
            Lrhs(j)=-g_uc(hL(j));
        end
        mat=[up_left_mat, H_u_mat';  H_u_mat, zeros(nLagrange)];
        rhs=[upper_rhs;Lrhs];
        a=diag(mat);a(n_g+2:end)=1;
        warning off;imat=inv(diag(a.^-1)*mat)*diag(a.^-1);warning on;
        sol=imat*rhs;
        %sol = mat\rhs;       % sometimes too ill-conditioned to work
        g=sol(1:n_g)+sol(n_g+1)+g_uc;
        g(hL)=0;
        Lset=sol(n_g+2:end);
        sim=X*g;
        if ireal==0
            sigma_old=sigma;
            sigma=sqrt((y-sim)'*(y-sim)/(length(y)-n_g+nLagrange-1));
            sigma=min(sigma,sigma_max);  %DAB
            disp(sprintf(['iteration %i: sigma = %8.3g, number of Lagrange ' ...
                      'multipliers %i'],[iter,sigma,nLagrange]));
        end
            %disp(sprintf('iteration %i: number of Lagrange multipliers %i',[iter,nL]));
            figure(1)
            set(gcf,'name',sprintf('Realization %i',ireal+1));
            subplot(3,1,1)
            handle=plot(t,y+me,'-r',t,sim,'-k');
            set(handle(1),'markersize',2);
            xlabel('t [hr]');
            legend('meas.','sim.');
            title(sprintf('output after iteration %i',iter));
            subplot(3,1,2)
            handle=plot([0:n_g-1]*dt,g,'k');
            title(sprintf('transfer function after iteration %i',iter));
            xlabel('\tau [hr]');
            drawnow
        hLold=hL;
        % set of entries that need Lagrange multiplier
        hLadd=ii(g<0);
        % remove entries that don't need a Lagrange multiplier anymore
        hLrem=hL(Lset>0);
        hL=hL(~ismember(hL,hLrem));
        hL=union(hL,hLadd);
        % DAB remove h(1)=0 requirement:
        % if (isempty(hL)), hL=1; elseif(hL(1)~=1), hL=[1;hL]; end 
        nLagrange=length(hL);
        if (isempty(setdiff(hLold,hL)) & isempty(setdiff(hL,hLold))) % | abs(sigma-sigma_old)/sigma<0.01
            % plot and save realization if converged
            ireal=ireal+1;
            t_g=dt*[0:n_g-1]';
            g_all(:,ireal) = g;
            subplot(3,1,3)
            hand=plot(t_g,mean(g_all(:,1:ireal),2),'r');
            set(hand,'linewidth',1.5);
            xlabel('\tau [hr]');
            hold on
            plot(t_g,prctile(g_all(:,1:ireal),[10 90],2),'b');
            plot(t_g,[min(g_all(:,1:ireal),[],2),max(g_all(:,1:ireal),[],2)],'b:');
            legend('mean','10%','90%','min','max');
            hold off
            break
        end
    end     % iterating until Lagrange multiplier set converges
end   % ensemble of realizations loop

% Re-estimate the covariance function

old_cov=cov;

% Cirpka's original (more or less)
theta_old=theta
theta=exp(fminsearch(@(lntheta) sumprob(g_all,nreal,n_g,lntheta),log(theta)));
g=g_all(:,1);
save([prefix '_transfer_func_condreal.mat'],'t_g','g','g_all','theta','sigma');
cov=zeros(1,length(g));
cov=[n_g:-1:1]*dt*theta;
%cov(1:n_g)=(theta/(n_g-1))*(n_g-1:-1:0);
size(cov)
% Get the actual cov of the current kernel:
if ~strcmp(method,'cirpka')
    g_mean=mean(g_all(:,1:ireal),2);
    g_mean=[g_mean; zeros(size(g_mean)) ]; 
    h_fft=fft(g_mean');
    cov=(1/length(g))*ifft( h_fft.*conj(h_fft) ); 
    cov=cov(1:length(g)); 
%    cov=xcorr(h_mean',h_mean','unbiased');
%    cov=cov(length(h_mean/2)+1:length(h_mean/2)+n_g);
    actual=cov;

    cov=max(0,cov);
    int_cov=dt*sum(cov);                        
    theta=cov(1);
    var_exp=var(mean(g_all(:,1:ireal),2));
    corr_time = min(n_g*dt, 2*int_cov/theta); % this is twice the real corr time for the triangle:
    theta=max(theta, 2*int_cov/corr_time);
    n_corr_time=ceil(corr_time/dt);
    exp_corr_time=int_cov/theta    % display the experimental correlation time
end
% Get a linear approximation of the actual kernel:
if strcmp(method,'linear')
    n_h_buffer=ceil(0.5*n_corr_time);
    cov=zeros(1,n_g)
    %cov=zeros(1,n_h_buffer+n_corr_time);
    cov(1:n_corr_time)=(theta/(n_corr_time-1))*(n_corr_time-1:-1:0);
    %n_g=n_h_buffer+n_corr_time;
end    

int_cov=dt*sum(cov); 
length_comp=min(length(cov), length(old_cov));
% This has changed a lot: one needs to scale the covariance change to total
% cov and signal to noise ratio.  Just a series of guesses.
rel_cov_change = sqrt(dt*sum((cov(1:length_comp)-old_cov(1:length_comp)).^2))/int_cov/(sigma/max(y))

figure(3)
plot(dt*(0:n_corr_time-1),(theta/(n_corr_time-1))*(n_corr_time-1:-1:0),'+-');
hold on
if ~strcmp(method,'cirpka') 
    plot(dt*(0:(length(actual)-1)),actual,'o')
end
legend('Autocovariance of g(t)','Linear approx.')
xlabel('Time lag (hr)'); ylabel('Autocovariance (1/hr^2)')
hold off

g_mean=mean(g_all(:,1:ireal),2);
L_2=sqrt(dt*sum((g_mean-kernel(1:size(g_mean,1))).^2))

end  % loop that goes until covariance function converges
toc

k=kernel(1:length(g));
kfft=[k; zeros(size(k))];
blah=fft(kfft');
cov_k=(1/length(g))*ifft(blah.*conj(blah));
cov_k=cov_k(1:length(g));
T_k=(dt/cov_k(1))*sum(cov_k)   % Sort of arbitrarily defined correlation time of real kernel


%Some stats:
trim_time=min(dt*n_g,4000);                  % DAB pick a trim time,    
n_trim = min(n_g,1+ceil(trim_time/dt));    % DAB Only use the first "trim" data points for fitting ADE
h_trim=g_mean(1:n_trim);
time=dt*(0:length(h_trim)-1)'; time(1)=1e-10;
m_0=dt*sum(h_trim)
m_1=(dt/m_0)*sum(time.*h_trim)
m_2=(dt/m_0)*sum(((time-m_1).^2).*h_trim) 
RMSE=sqrt(mean((X*g_mean-y).^2))

% plot final results
figure(222)
plot(dt*(1:300),kernel(1:300),'k','LineWidth',1.5)
hold on
plot(dt*(1:length(x)),x,'b')
plot(dt*(1:length(y)),y,'r')
xlabel('Time'); 
hold off

figure(2)
plot(t_g,kernel(1:n_g),'k-','LineWidth',1.5)
hold on
plot(t_g,g_mean,'r');
xlabel('\tau [hr]'); ylabel('g(\tau) [1/hr]');
plot(t_g,prctile(g_all(:,1:ireal),10,2),'b');
plot(t_g,prctile(g_all(:,1:ireal),90,2),'b');
%axis([0 40 0 .15]);
%txt ={['g(t) stats:'],['kernel mass: ' num2str(m_0)],['mean (hr):   ' num2str(m_1)],['Var (hr^2): ' num2str(var_exp)]};
%text(12,.25,txt)
hold off

figure(22)
semilogy(t_g,mean(g_all(:,1:ireal),2),'r');
xlabel('\tau [hr]'); ylabel('g(\tau) [1/hr]');
hold on
plot(t_g,prctile(g_all(:,1:ireal),10,2),'b');
plot(t_g,prctile(g_all(:,1:ireal),90,2),'b');
plot(dt*((1:length(kernel))-1),kernel,k)
axis([0 24 1e-5 1]);
hold off

function lnpsum = sumprob(h_all,nreal,n_h,lntheta)
% probability of h according to prior statistics
theta=exp(lntheta);
lnpsum=0;
for ireal=1:nreal
    num_nonzero=sum(h_all(:,ireal)>0);
    ng =size(h_all,1);
    num_zeros = ng - num_nonzero;
    lnp_i= -num_nonzero/2*log(4*pi*theta)-(ng-1)/2*log(1);
    % So, addly enough, if you look at this section, it does nothing, so
    % DAB commented it out:
    % loop over all zero entries
    % im=1;
    % while (h_all(im,ireal)>0)
    %     im=im+1
    % end
    % ip=im;  % This is the index of the first zero entry
    % for jj=1:num_zeros-1
    %     % determine next entry
    %     ip=ip+1;
    %     while (h_all(ip,ireal)>0)
    %         ip=ip+1;
    %     end
    % end

    % loop over all entries
    for jj=1:ng-1
        lnp_i = lnp_i - (h_all(jj+1,ireal)-h_all(jj,ireal))^2/(4*theta);
    end
    lnpsum=lnpsum-lnp_i;
end
end
