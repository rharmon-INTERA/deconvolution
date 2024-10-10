clear all
load("4jun19_medQ.mat");  % 4 fluid BTCs 
time = zero_time_R4_US;
n = length(time);  % Change length of all vectors if desired
cout_1A = (R4_US_cond_no_bg(1:n).*0.3);
cout_1B = (R4_DS_cond_no_bg(1:n).*0.3); 
cout_2A = (R3_US_cond_no_bg(1:n).*0.3);
cout_2B = (R3_DS_cond_no_bg(1:n).*0.3); 
%%%%%%%%%%%%%%%%%%%
load('jun4_R4_US.mat');   % Bulk BTC at 1A (I think)
time_imm=times_decimal_noBg % Different for each bulk BTC!
in=cout_1A(1:n);
out=interp1(time_imm,cond_noBg,time); out(out<0)=0;
save ('medQ_R1A_MIM.mat', 'time', 'in', 'out')

figure(44)
plot(time,in,'o-')
hold on 
plot(time,out,'+-')
%axis([-1 5 -1 20])
legend
hold off

%%%%%%%%%%%%%%%%%%%
load('jun4_R4_DS.mat');   % Bulk BTC at 1B (I think)
time_imm=times_decimal_noBg % Different for each bulk BTC!
in=cout_1B(1:n);
out=interp1(time_imm,cond_noBg,time); out(out<0)=0;
save ('medQ_R1B_MIM.mat', 'time', 'in', 'out')
%%%%%%%%%%%%%%%%%%%
load('jun4_R3_US.mat');   % Bulk BTC at 2A (I think)
time_imm=times_decimal_noBg % Different for each bulk BTC!
in=cout_2A(1:n);
out=interp1(time_imm,cond_noBg,time); out(out<0)=0;
save ('medQ_R2A_MIM.mat', 'time', 'in', 'out')
%%%%%%%%%%%%%%%%%%%%
load('jun4_R3_DS.mat');   % Bulk BTC at 2B (I think)
time_imm=times_decimal_noBg % Different for each bulk BTC!
in=cout_2B(1:n);
out=interp1(time_imm,cond_noBg,time); out(out<0)=0;
save ('medQ_R2B_MIM.mat', 'time', 'in', 'out')

figure(44)
plot(time,in,'o-')
hold on 
plot(time,out,'+-')
%axis([-1 5 -1 20])
legend
hold off


