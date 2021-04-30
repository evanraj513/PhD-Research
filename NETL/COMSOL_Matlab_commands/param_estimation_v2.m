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

Noise is added to the data, as the major difference from v1. 

%}

clear data_w_noise
% 
global std19 data_w_noise
% model = mphopen('Evan_segmented_faraday_v8');
std19 = model.study('std19');

% q_true = 10/6;
% data = comsol_run(q_true);

data_w_noise = {};

for a = 1:length(data)
    %% Adding noise to data
    noise = randn(size(data(a).d1));
    data_new = data(a).d1 + noise*mean(data(a).d1);
    data_w_noise{end+1} = data_new; %% Appending noise
end

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
    'FinDiffRelStep',1e-3,... %% Makes delta = v.*sign(x).*max(abs(x),TypicalX) with v = FinDiffRelStep
    'TolX',1e-6);
% 'PlotFcn',{'optimplotfval'},...

[q_star,resnorm,residual,exitflag,output] = lsqnonlin(@obj_fun,q_1,lb,ub,lsqnonlin_options);

% [q_star,resnorm,residual,exitflag,output] = additive_noise_pe(0.1);

function [q_star,resnorm,residual,exitflag,output] = additive_noise_pe(noise_level)
    global std19 data_w_noise data model
    % model = mphopen('Evan_segmented_faraday_v8');
    std19 = model.study('std19');

    data_w_noise = {};

    for a = 1:length(data)
        %% Adding noise to data
        noise = randn(size(data(a).d1));
        data_new = data(a).d1 + noise*mean(data(a).d1)*noise_level;
        data_w_noise{end+1} = data_new; %% Appending noise
    end

    q_1 = 10/6+0.1;
    
    lb = [0];
    ub = [1000];

    lsqnonlin_options = optimset('Display','iter-detailed',...
    'TolFun',1e-6,...
    'MaxIter',15,...
    'Algorithm','trust-region-reflective',... %% Default is trust-region-reflective, other is levenberg-marquardt
    'FinDiffRelStep',1e-3,... %% Makes delta = v.*sign(x).*max(abs(x),TypicalX) with v = FinDiffRelStep
    'TolX',1e-6);
    % 'PlotFcn',{'optimplotfval'},...
    [q_star,resnorm,residual,exitflag,output] = lsqnonlin(@obj_fun,q_1,lb,ub,lsqnonlin_options);
end

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

function [err_3] = obj_fun(q)
    global data_w_noise
    
    disp('Current q value: ')
    disp(q)

    ret = comsol_run(q);
    
    pdJx = ret(1);
    pdJy = ret(2);
    pdJz = ret(3);
    pdV = ret(4);
   
    diff = [norm(data_w_noise{1} - pdJx.d1)/norm(data_w_noise{1}),...
            norm(data_w_noise{2} - pdJy.d1)/norm(data_w_noise{2}),...
            norm(data_w_noise{3} - pdJz.d1)/norm(data_w_noise{3}),... 
            norm(data_w_noise{4} - pdV.d1)/norm(data_w_noise{4})];
    err = norm(diff); %% l2 norm of l2 norm of differences, scaled by l2 norm of data
    
    err_2 = [((data_w_noise{1} - pdJx.d1)/norm(data_w_noise{1}))';...
        ((data_w_noise{2} - pdJy.d1)/norm(data_w_noise{2}))';...
        ((data_w_noise{3} - pdJz.d1)/norm(data_w_noise{3}))';...
        ((data_w_noise{4} - pdV.d1)/norm(data_w_noise{4}))'];
    % Naive approximation to relative error, all appended together into vector, 
    
    err_3 = [((data_w_noise{1} - pdJx.d1))';...
        ((data_w_noise{2} - pdJy.d1))';...
        ((data_w_noise{3} - pdJz.d1))';...
        ((data_w_noise{4} - pdV.d1))'];
    % Absolute error between points
    
    err_4 = [((data_w_noise{1} - pdJx.d1)/length(data_w_noise{1}))';...
        ((data_w_noise{2} - pdJy.d1)/length(data_w_noise{2}))';...
        ((data_w_noise{3} - pdJz.d1)/length(data_w_noise{3}))';...
        ((data_w_noise{4} - pdV.d1)/length(data_w_noise{4}))'];
    % On the order of the grid norm error 
end