%{
This script completes the parameter estimation problem. It is a stand-alone
script. It will run open the model, extract data, add a varying amount of
noise, perform the parameter estimation problem, and then return an array
of [noise level, returned parameter estimate, # of iters, resnorm^2]. 

If running for the first time, uncomment global, model std19, and data. 
Also, copy the ModelUtil before running to display a progress bar. 
%}

%% Setting up model and data
% global model std19 data
% 
% model = mphopen('Evan_segmented_faraday_v8');
% std19 = model.study('std19');
% 
% q_true = 10/6; %% True parameter value
% data = comsol_run(q_true); %% Fabricated 'True' Data

%% Running PE Additive

% q_initial = q_true*1.1; %% Initial parameter guess

% noise_all = 0:0.2:3;
% pe_est_all = zeros(size(noise_all));
% iters_all = zeros(size(noise_all));
% func_count_all =  zeros(size(noise_all));
% resnorm_all = zeros(size(noise_all));
% 
% ticker = 1;
% for a = noise_all
%     disp('** Currently Testing Noise-level:**')
%     disp(a)
%     [q_star,resnorm,residual,exitflag,output] = additive_noise_pe(a,q_initial);
%     pe_est_all(ticker) = q_star;
%     iters_all(ticker) = output.iterations;
%     func_count_all(ticker) = output.funcCount;
%     resnorm_all(ticker) = resnorm;
%     ticker = ticker+1;
% end

%% Running PE Multiplicative

% noise_all_mult = 0:0.2:3;
% pe_est_all_mult = zeros(size(noise_all_mult));
% iters_all_mult = zeros(size(noise_all_mult));
% func_count_all_mult =  zeros(size(noise_all_mult));
% resnorm_all_mult = zeros(size(noise_all_mult));
% 
% ticker_mult = 1;
% tic
% for a = noise_all_mult
%     disp('** Currently Testing Noise-level:**')
%     disp(a)
%     [q_star,resnorm,residual,exitflag,output] = multiplicative_noise_pe(a,q_initial);
%     pe_est_all_mult(ticker_mult) = q_star;
%     iters_all_mult(ticker_mult) = output.iterations;
%     func_count_all_mult(ticker_mult) = output.funcCount;
%     resnorm_all_mult(ticker_mult) = resnorm;
%     ticker_mult = ticker_mult+1;
% end
% toc
% disp('This loop took')
% disp(toc)

%% Plotting Additive
x = noise_all;
abs_err_add = abs(pe_est_all - q_true);
rel_err = abs_err_add/q_true;
a = 1;

figure(a);
plot(x,rel_err,'-x');
title('Relative Error vs. Additive Noise Level');
xlabel('Noise Level');
ylabel('Relative Error');

% figure(a+1);
% plot(x,iters_all,'-x');
% title('# of Iterations vs. Additive Noise Level');
% xlabel('Noise Level');
% ylabel('# of iterations');

%% Plotting Multiplicative
% noise_all_mult = noise_all_mult(1:12);
% pe_est_all_mult = pe_est_all_mult(1:12);
% iters_all_mult = iters_all_mult(1:12);
% func_count_all_mult =  func_count_all_mult(1:12);
% resnorm_all_mult = resnorm_all_mult(1:12);

x = noise_all_mult;
a = 2;

figure(a);
abs_err_add = abs(pe_est_all_mult - q_true);
rel_err = abs_err_add/q_true;
plot(x,rel_err,'-x');
title('Relative Error vs. Multplicative Noise Level');
xlabel('Noise Level');
ylabel('Relative Error');

% figure(a+1);
% plot(x,iters_all_mult,'-x');
% title('# of Iterations vs. Additive Noise Level');
% xlabel('Noise Level');
% ylabel('# of iterations');

%% Functions

function [q_star,resnorm,residual,exitflag,output] = additive_noise_pe(noise_level,initial_guess)
    global data_w_noise data
    
    data_w_noise = {};

    for a = 1:length(data)
        %% Adding noise to data
        noise = randn(size(data(a).d1));
        data_new = data(a).d1 + noise*mean(data(a).d1)*noise_level;
        data_w_noise{end+1} = data_new; %% Appending noise
    end

    q_1 = initial_guess;
    
    lb = [0];
    ub = [1000];

    lsqnonlin_options = optimset('Display','iter-detailed',...
    'TolFun',1e-6,...
    'MaxIter',25,...
    'Algorithm','trust-region-reflective',... %% Default is trust-region-reflective, other is levenberg-marquardt
    'FinDiffRelStep',1e-3,... %% Makes delta = v.*sign(x).*max(abs(x),TypicalX) with v = FinDiffRelStep
    'TolX',1e-6);
    % 'PlotFcn',{'optimplotfval'},...
    [q_star,resnorm,residual,exitflag,output] = lsqnonlin(@obj_fun,q_1,lb,ub,lsqnonlin_options);
end

function [q_star,resnorm,residual,exitflag,output] = multiplicative_noise_pe(noise_level,initial_guess)
    global data_w_noise data
    
    data_w_noise = {};

    for a = 1:length(data)
        %% Adding noise to data
        noise = noise_level*randn(size(data(a).d1));
        data_new = data(a).d1.*(1+noise);
        data_w_noise{end+1} = data_new; %% Appending noise
    end

    q_1 = initial_guess;
    
    lb = [0];
    ub = [1000];

    lsqnonlin_options = optimset('Display','iter-detailed',...
    'TolFun',1e-6,...
    'MaxIter',25,...
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

function [err_4] = obj_fun(q)
    global data_w_noise
    
    disp('Current q value: ')
    disp(q)

    ret = comsol_run(q);
    
    pdJx = ret(1);
    pdJy = ret(2);
    pdJz = ret(3);
    pdV = ret(4);
   
%     diff = [norm(data_w_noise{1} - pdJx.d1)/norm(data_w_noise{1}),...
%             norm(data_w_noise{2} - pdJy.d1)/norm(data_w_noise{2}),...
%             norm(data_w_noise{3} - pdJz.d1)/norm(data_w_noise{3}),... 
%             norm(data_w_noise{4} - pdV.d1)/norm(data_w_noise{4})];
%     err = norm(diff); %% l2 norm of l2 norm of differences, scaled by l2 norm of data
%     
%     err_2 = [((data_w_noise{1} - pdJx.d1)/norm(data_w_noise{1}))';...
%         ((data_w_noise{2} - pdJy.d1)/norm(data_w_noise{2}))';...
%         ((data_w_noise{3} - pdJz.d1)/norm(data_w_noise{3}))';...
%         ((data_w_noise{4} - pdV.d1)/norm(data_w_noise{4}))'];
%     % Naive approximation to relative error, all appended together into vector, 
%     
%     err_3 = [((data_w_noise{1} - pdJx.d1))';...
%         ((data_w_noise{2} - pdJy.d1))';...
%         ((data_w_noise{3} - pdJz.d1))';...
%         ((data_w_noise{4} - pdV.d1))'];
    % Absolute error between points
    
    err_4 = [((data_w_noise{1} - pdJx.d1)/length(data_w_noise{1}))';...
        ((data_w_noise{2} - pdJy.d1)/length(data_w_noise{2}))';...
        ((data_w_noise{3} - pdJz.d1)/length(data_w_noise{3}))';...
        ((data_w_noise{4} - pdV.d1)/length(data_w_noise{4}))'];
    % On the order of the grid norm error 
end


























