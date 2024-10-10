%function deconv_olaf_condsim
clear
%%%%%%%%%%  Some reach details %%%%%%%%%%%%%%%%%%%%%
prefix=['medQ_R1A_MIM'];
load([prefix '.mat']);  % three vectors of equal length: time, in, out
dx=35;                 % m downstream of the upstream
%%%%%%%%%%  Some user-defined numerical details %%%%%%%%%%%%%%%%%%%%%
% slope of the linear variogram
theta=1e-4;       %  A first guess applied linearly to corr_time:
corr_time=10;   %  Only applies if less than the total h(t) length
% standard deviation of epistemic error (initial guess)
sigma = .231;
sigma_max= .1;   % DAB places a forced maximum on iterated sigma
% length of transfer-function vector (dt remains the same)
n_h=1600;  
% number of realizations
nreal=10;  % Make 50 for nice final plots - takes a while
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rand('state',sum(100*clock));
% input signal
x=in;
x(x<0)=0;
% output signal
y=out;
t=time;
% time increment
dt=time(2)-time(1);
corr_time = min(n_h*dt,corr_time);  % DAB
n_corr_time=ceil(corr_time/dt);     % DAB
cov=zeros(1,n_h);                   % First guess
cov(1:n_corr_time)=(theta/n_corr_time)*(n_corr_time:-1:1);  % DAB
% construction of Jacobian (convolution matrix)
c=dt*x;
r=dt*zeros(1,n_h);
J=toeplitz(c,r);

theta_old=0;

while(abs(theta_old-theta)/theta>0.02)
% construction of generalized covariance matrix (here time in units of "time")
c=[n_h:-1:1]*dt*theta;   % Olaf's
c(1:n_corr_time)=[n_corr_time:-1:1]*dt*theta;  % DAB's   approx linear cov function
c(1:length(cov))=cov;                          % DAB     actual cov function
%pause
Q=toeplitz(c);
C=chol(Q);
iQ=inv(Q);

% vector of indices
ii=[1:n_h]';

% Best estimate
% initialization of constraints
 hL=[];
 nL=0;
 iter=0;
 while iter<40
       iter=iter+1;
       % construction of unconstrained matrix
       JRJ=J'*J/sigma^2;
       u=ones(n_h,1);
       umat=[JRJ+iQ,JRJ*u;u'*JRJ,u'*JRJ*u];
       urhs=[J'*y/sigma^2;u'*J'*y/sigma^2];
       % matrix related to the Lagrange multipliers
       Lmat=zeros(n_h+1,nL);
       Lrhs=zeros(nL,1);
       for j=1:nL
           Lmat(hL(j),j)=1;
           Lmat(n_h+1,j)=1;
           Lrhs(j)=0;
       end
       mat=[umat,Lmat;Lmat',zeros(nL)];
       rhs=[urhs;Lrhs];
       a=diag(mat);a(n_h+2:end)=1;
       warning off;imat=inv(diag(a.^-1)*mat)*diag(a.^-1);warning on;
       sol=imat*rhs;
       h_be=sol(1:n_h)+sol(n_h+1);
       h_be(hL)=0;
       nu=sol(n_h+2:end);
       sim=J*h_be;
       sigma=sqrt((y-sim)'*(y-sim)/(length(y)-n_h+nL-1));
       sigma=min(sigma,sigma_max);  %DAB
       disp(sprintf(['iteration %i: sigma = %8.3g, number of Lagrange ' ...
                     'multipliers %i'],[iter,sigma,nL]));
       hLold=hL;
       % set of entries that need Lagrange multiplier
       hLadd=ii(h_be<0);
       % remove entries that don't need a Lagrange multiplier anymore
       hLrem=hL(nu>0);
       hL=hL(~ismember(hL,hLrem));
       hL=union(hL,hLadd);
       % DAB remove h(1)=0 requirement:
       %  if (isempty(hL)), hL=1; elseif(hL(1)~=1), hL=[1;hL]; end
       nL=length(hL);
       if isempty(setdiff(hLold,hL)) & isempty(setdiff(hL,hLold)), break, end
end    

% initialize sum of h and sum of h squared
h_all  = zeros(n_h,nreal);

%figure(1);clf;
% loop over all realizations
ireal=0;
while ireal<nreal   % DAB changed to reject non-converged realizations
    % unconditional realiuzation
    h_uc=C'*randn(n_h,1);
    % measurement error
    me = sigma*randn(size(y));
    % initialization of constraints
    hL=[];
    nL=0;
    iter=0;
    while iter<50
        iter=iter+1;
        % construction of unconstrained matrix
        JRJ=J'*J/sigma^2;
        u=ones(n_h,1);
        umat=[JRJ+iQ,JRJ*u;u'*JRJ,u'*JRJ*u];
        urhs=[J'*(y+me)/sigma^2-JRJ*h_uc;u'*J'*(y+me)/sigma^2-u'*JRJ*h_uc];
        % matrix related to the Lagrange multipliers
        Lmat=zeros(n_h+1,nL);
        Lrhs=zeros(nL,1);
        for j=1:nL
            Lmat(hL(j),j)=1;
            Lmat(n_h+1,j)=1;
            Lrhs(j)=-h_uc(hL(j));
        end
        mat=[umat,Lmat;Lmat',zeros(nL)];
        rhs=[urhs;Lrhs];
        a=diag(mat);a(n_h+2:end)=1;
        warning off;imat=inv(diag(a.^-1)*mat)*diag(a.^-1);warning on;
        sol=imat*rhs;
        h=sol(1:n_h)+sol(n_h+1)+h_uc;
        h(hL)=0;
        nu=sol(n_h+2:end);
        sim=J*h;
        disp(sprintf('iteration %i: number of Lagrange multipliers %i',[iter,nL]));
       figure(1)
       set(gcf,'name',sprintf('Realization %i',ireal+1));
       subplot(3,1,1)
       handle=plot(t,y+me,'-r',t,sim,'-k');
       set(handle(1),'markersize',2);
       xlabel('t [hr]');
       legend('meas.','sim.');
       title(sprintf('output after iteration %i',iter));
       subplot(3,1,2)
       handle=plot([0:n_h-1]*dt,h,'k');
       title(sprintf('transfer function after iteration %i',iter));
       xlabel('\tau [hr]');
       drawnow
        hLold=hL;
        % set of entries that need Lagrange multiplier
        hLadd=ii(h<0);
        % remove entries that don't need a Lagrange multiplier anymore
        hLrem=hL(nu>0);
        hL=hL(~ismember(hL,hLrem));
        hL=union(hL,hLadd);
        % DAB remove h(1)=0 requirement:
        %  if (isempty(hL)), hL=1; elseif(hL(1)~=1), hL=[1;hL]; end 
        nL=length(hL);
        if isempty(setdiff(hLold,hL)) & isempty(setdiff(hL,hLold))
            % plot and save realization if converged
            ireal=ireal+1;
            t_h=dt*[0:n_h-1]';
            h_all(:,ireal) = h;
            subplot(3,1,3)
            hand=plot(t_h,mean(h_all(:,1:ireal),2),'r');
            set(hand,'linewidth',1.5);
            xlabel('\tau [hr]');
            hold on
            plot(t_h,prctile(h_all(:,1:ireal),[10 90],2),'b');
            plot(t_h,[min(h_all(:,1:ireal),[],2),max(h_all(:,1:ireal),[],2)],'b:');
            legend('mean','10%','90%','min','max');
            hold off
            break
        end
    end    
    % t_h=dt*[0:n_h-1]';
    % h_all(:,ireal) = h;
    % subplot(3,1,3)
    % hand=plot(t_h,mean(h_all(:,1:ireal),2),'r');
    % set(hand,'linewidth',1.5);
    % xlabel('\tau [hr]');
    % hold on
    % plot(t_h,prctile(h_all(:,1:ireal),[10 90],2),'b');
    % plot(t_h,[min(h_all(:,1:ireal),[],2),max(h_all(:,1:ireal),[],2)],'b:');
    % legend('mean','10%','90%','min','max');
    % hold off
end

theta_old=theta
theta=exp(fminsearch(@(lntheta) sumprob(h_all,nreal,n_h,lntheta),log(theta)));
h=h_be;
save([prefix '_transfer_func_condreal.mat'],'t_h','h','h_all','theta','sigma');

h_mean=mean(h_all(:,1:ireal),2);                    % DAB 
h_fft=fft([h_mean; zeros(size(h_mean))] );          % DAB  Zero padding for fft
cov=(1/length(h))*ifft( h_fft.*conj(h_fft) );       % DAB
cov=cov(1:length(h_mean));                          % DAB  this is E(h(t)h(t_s))
%cov=cov-(mean(h_mean))^2;                           % DAB  For neg. of semivar
int_cov=dt*sum(cov);                                % DAB
theta=cov(1)                                        % DAB
var_exp=var(h_mean)                                 % DAB
corr_time = min(n_h*dt, 2*int_cov/theta)            % DAB
theta=max(theta, 2*int_cov/corr_time)               % DAB
n_corr_time=ceil(corr_time/dt);                     % DAB

figure(3)
plot(dt*(0:(length(h)-1)),cov,'o')
hold on 
plot(dt*(0:n_corr_time-1),(theta/(n_corr_time-1))*(n_corr_time-1:-1:0),'+-');
legend('C(s)=<g(t+s)g(t)>','Linear approx.')
xlabel('Time lag s (hr)'); ylabel('C(s) (1/hr^2)')
hold off

end

%Some stats:
trim_time=min(dt*n_h,16);
n_trim = min(n_h,1+ceil(trim_time/dt));       % DAB Only use the first "trim" data points for fitting ADE
h_trim=h_mean(1:n_trim);
time=dt*(0:length(h_trim)-1)'; time(1)=1e-10;
m_0=dt*sum(h_trim)
m_1=(dt/m_0)*sum(time.*h_trim)
var_exp=(dt/m_0)*sum(((time-m_1).^2).*h_trim) 
RMSE=sqrt(mean((J*h-y).^2))

%  Generate an Inverse Gaussian with same parameters
time_inv=linspace(0,(n_trim-1)*dt,1000); time_inv(1)=1e-10;
h_inv=interp1(time,h_trim,time_inv);
v=dx/m_1;           % m/hr
Disp=var_exp*v^3/dx/2;    % m^2/hr

% or best-fit using LM

V = optimvar('V',1);
D = optimvar('D',1);
fun = (dx./sqrt(4*pi*D*time_inv.^3)).*exp((dx-V*time_inv).^2./(-4*D*time_inv));
obj = sum((fun - h_inv).^2);
lsqproblem = optimproblem("Objective",obj);
x0.V=v;
x0.D=Disp;
[solution,fval] = solve(lsqproblem,x0)

v=solution.V; Disp=solution.D;
tplot=linspace(0,1,500); dt_plot=tplot(2)-tplot(1);

InvG = (dx./sqrt(4*pi*Disp*tplot.^3)).*exp((dx-v*tplot).^2./(-4*Disp*tplot));
InvG(1)=0;


% plot final results
figure(2)
plot(t_h,mean(h_all(:,1:ireal),2),'r');
xlabel('\tau [hr]'); ylabel('g(\tau) [1/hr]');
hold on
plot(t_h,prctile(h_all(:,1:ireal),10,2),'b');
plot(t_h,prctile(h_all(:,1:ireal),90,2),'b');
%plot(tplot,InvG,'b--')
axis([0 24 0 1]);
txt ={['g(t) stats:'],['kernel mass: ' num2str(m_0)],['mean (hr):   ' num2str(m_1)],['Var (hr^2): ' num2str(var_exp)]};
text(12,.5,txt)
%txt ={['ADE parameters:'],['v (m/hr): ' num2str(v)],['D (m^2/hr): ' num2str(Disp)]};
%text(.25,7,txt)
hold off

figure(22)
semilogy(t_h,mean(h_all(:,1:ireal),2),'r');
xlabel('\tau [hr]'); ylabel('g(\tau) [1/hr]');
hold on
plot(t_h,prctile(h_all(:,1:ireal),10,2),'b');
plot(t_h,prctile(h_all(:,1:ireal),90,2),'b');
%plot(tplot,InvG,'b--')
axis([0 24 1e-5 1]);
hold off

function lnpsum = sumprob(h_all,nreal,n_h,lntheta)
% probability of h according to prior statistics
theta=exp(lntheta);
lnpsum=0;
for ireal=1:nreal
    nnz=sum(h_all(:,ireal)>0);
    ng =size(h_all,1);
    nz = ng-nnz;
    lnp_i= -nnz/2*log(4*pi*theta)-(ng-1)/2*log(1);
    % loop over all zero entries
    im=1; while (h_all(im,ireal)>0), im=im+1;end
    ip=im;
    for jj=1:nz-1
        % determine next entry
        ip=ip+1;while (h_all(ip,ireal)>0), ip=ip+1;end
    end
    % loop pver all entries
    for jj=1:ng-1
        lnp_i = lnp_i - (h_all(jj+1,ireal)-h_all(jj,ireal))^2/(4*theta);
    end
    lnpsum=lnpsum-lnp_i;
end
end

%function resid = ADE_obj(vD)
%global dx
%v=vD(1);
%Disp=vD(2);
%InvG = (dx./sqrt(2*pi*Disp*time.^3)).*exp((dx-v*time).^2./(-2*Disp*time));
%InvG(1)=0;
%resid=(InvG-h_mean).^2
%end




