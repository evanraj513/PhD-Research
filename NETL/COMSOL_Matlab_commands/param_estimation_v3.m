%{
This script completes the parameter estimation problem. Depending
on when running the script, it can be adapted by uncommenting or commenting
several lines, namely model, std19, q_true, and data

The sol'n is assumed to be [Jf_i, V], whose values are given for a
particular parameter set via FMGRES within COMSOL. 

Major difference from v1, v2 is that error will now have to be extracted
from COMSOL's built-in integration, so that the error given is now a
grid-norm error rather than the simpler l2 error (as done in v1, v2).
However, this requires (FOR NOW) that the user update the data file within
the COMSOL GUI, and not done directly here. 

No noise is being added to the data yet, it has been taken directly from
COMSOL data extraction. The data file currently being used is named: 
    opt_test_data_1.csv
Note that this was run with q = mob_e = 10/6


The minimization solver is given by lsqnonlin. 
%}

clear data_w_noise

global std19 
% model = mphopen('Evan_segmented_faraday_v8');
std19 = model.study('std19');

q_1 = 10/6+0.1;

% options = optimset('Display','iter','MaxIter',10);
% [q_star,err] = fminsearch(@obj_fun,10/6+0.1,options);

% A = [];
% b = [];
% Aeq = [];
% beq = [];
lb = [0];
ub = [1000];
% 
% [q_star,err] = fmincon(@obj_fun,10/6+0.1,A,b,Aeq,beq,lb,ub,[],options);

lsqnonlin_options = optimset('Display','iter-detailed',...
    'TolFun',1e-5,...
    'MaxIter',15,...
    'TolX',1e-6);
% 'PlotFcn',{'optimplotfval'},...
[q_star,resnorm,residual,exitflag,output] = lsqnonlin(@obj_fun,q_1,lb,ub,lsqnonlin_options);

function [] = comsol_run(q)   
    global std19 
    for a = 1:length(q)
        std19.feature('Par_sweep').setIndex('plistarr',q(a),a-1);
    end
    std19.run
    
end

function [err] = obj_fun(q)
    global model
    disp('Currently testing q: ')
    disp(q)
    comsol_run(q);
    
    tbl = mphtable(model,'tbl1');
    
    err = tbl.data(2);
    
end