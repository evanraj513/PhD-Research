%{
This script completes the parameter estimation problem. Depending
on when running the script, it can be adapted by uncommenting or commenting
several lines, namely model, std19, q_true, and data

To run another model, simply open it in the command window and run the
script

The sol'n is assumed to be [Jf_i, V], whose values are given for a
particular parameter set via FMGRES within COMSOL. Set-up is done via the
function comsol_run. The data is then compared directly with each computed
solution in the obj_fun. The solver is given by lsqnonlin. 

%}

global std19
% model = mphopen('Evan_segmented_faraday_v8');
std19 = model.study('std19');

% q_true = 10/6;
% data = comsol_run(q_true);

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
    'TolFun',1e-6,...
    'MaxIter',15,...
    'Algorithm','trust-region-reflective',... %% Default is trust-region-reflective, other is levenberg-marquardt
    'FinDiffRelStep',1e-3,... %% Does this do more than set the initial step size?
    'TolX',1e-6);
% 'PlotFcn',{'optimplotfval'},...
[q_star,resnorm,residual,exitflag,output] = lsqnonlin(@obj_fun,q_1,lb,ub,lsqnonlin_options);

function [ret] = comsol_run(q)   
    global std19 model
    for a = 1:length(q)
        std19.feature('Par_sweep').setIndex('plistarr',q(a),a-1);
    end
    std19.run
    pdJx = mpheval(model,'ec.Jx','dataset','dset28');
    pdJy = mpheval(model,'ec.Jy','dataset','dset28');
    pdJz = mpheval(model,'ec.Jz','dataset','dset28');
    pdV = mpheval(model,'V','dataset','dset28');
    ret = [pdJx, pdJy, pdJz, pdV];
end

function [err_4] = obj_fun(q)
    global data
    
    disp('Current q value: ')
    disp(q)

    ret = comsol_run(q);
    
    pdJx = ret(1);
    pdJy = ret(2);
    pdJz = ret(3);
    pdV = ret(4);
    
    diff = [norm(data(1).d1 - pdJx.d1)/norm(data(1).d1),...
            norm(data(2).d1 - pdJy.d1)/norm(data(2).d1),...
            norm(data(3).d1 - pdJz.d1)/norm(data(3).d1),... 
            norm(data(4).d1 - pdV.d1)/norm(data(4).d1)];
       
    err = norm(diff);
    
    err_2 = [((data(1).d1 - pdJx.d1)/norm(data(1).d1))';...
        ((data(2).d1 - pdJy.d1)/norm(data(2).d1))';...
        ((data(3).d1 - pdJz.d1)/norm(data(3).d1))';...
        ((data(4).d1 - pdV.d1)/norm(data(4).d1))'];
    
    err_3 = [((data(1).d1 - pdJx.d1))';...
        ((data(2).d1 - pdJy.d1))';...
        ((data(3).d1 - pdJz.d1))';...
        ((data(4).d1 - pdV.d1))'];
    
    err_4 = [((data(1).d1 - pdJx.d1)/length(data(1).d1))';...
        ((data(2).d1 - pdJy.d1)/length(data(2).d1))';...
        ((data(3).d1 - pdJz.d1)/length(data(3).d1))';...
        ((data(4).d1 - pdV.d1)/length(data(4).d1))'];
    
    %% l2_norm(l2_norm(J^N_x - D_x) + l2_norm(J^N_y - D_y) + l2_norm(J^N_z - D_z) + l2_norm(V^N - d))
    
end