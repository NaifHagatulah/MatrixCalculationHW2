load('cgn_illustration.mat')
close all;
format long;

compare_gmres_cgn(A, b)

% Main script to compare GMRES and CGN
function compare_gmres_cgn(A, b)
    max_iter = 60;  % Maximum iterations
    
    % Arrays to store results
    gmres_residuals = zeros(max_iter, 1);
    cgn_residuals =   zeros(max_iter, 1);
    gmres_times =     zeros(max_iter, 1);
    cgn_times =       zeros(max_iter, 1);
    
    % Initial time
    gmres_start = tic;
    x_gmres = zeros(size(b));
    r0 = b - A*x_gmres;
    gmres_times(1) = toc(gmres_start);
    gmres_residuals(1) = norm(r0);
    
    cgn_start = tic;
    [x_cgn, ~] = CG(A, b, 1, true);
    
    cgn_times(1) = toc(cgn_start);
    cgn_residuals(1) = norm(A*x_cgn - b);

    % Main iteration loop
    for k = 2:max_iter
        % GMRES
        gmres_start = tic;
        [x_gmres, res_norm] = GMRES_opt(A, b, k);
        gmres_times(k) = toc(gmres_start);
        gmres_residuals(k) = res_norm;
        
        % CGN
        cgn_start = tic;
        [x_cgn, r_hist] = CG(A, b, k, true);
        cgn_times(k) =  toc(cgn_start);
        cgn_residuals(k) = r_hist(end);
    end
    % Plot results
    figure(1)
    semilogy(0:max_iter-1, gmres_residuals, 'b-', 'LineWidth', 1.5)
    hold on
    semilogy(0:max_iter-1, cgn_residuals, 'r-', 'LineWidth', 1.5)
    xlabel('Iteration')
    ylabel('||Ax - b||_2')
    legend('GMRES', 'CGN')
    title('Convergence of GMRES and CGN')
    grid on
    %saveas(gcf, "GMRES_CGN_Convergence.png")
    hold off
    
    figure(2)
    semilogy(gmres_times, gmres_residuals, 'b-', 'LineWidth', 1.5)
    hold on
    semilogy(cgn_times, cgn_residuals, 'r--', 'LineWidth', 1.5)
    xlabel('CPU-time (seconds)')
    ylabel('||Ax - b||_2')
    legend('GMRES', 'CGN')
    title('Computation time of GMRES and CGN')
    grid on
    saveas(gcf, "CPUtime.png")
    hold off
end




function [x, rhist] = CG(A,b,N,NE)
    x = zeros(size(b));
    if NE == true
        AT = A';
    else
        AT = 1;
    end
    r = AT*b;
    p = r;
    rhist = [];
    for k = 1:N
        rr = r'*r;
        Ap = AT*(A*p);
        alpha = rr/(p'*Ap);
        x = x + alpha*p;
        r = r - alpha*Ap;
        beta = (r'*r)/rr;
        p = r + beta*p;
        rhist = [rhist, norm(r)];
    end
end


function [x_approx, res_norm] = GMRES_opt(A,b, m)
    x_approx = zeros(length(b), 1);
    r0 =  b  - A*x_approx;
    [Q, H] = arnoldi(A,r0,m);     
    e1 = [1; zeros(m,1)];  
    z = (H\e1)*norm(b);
    x_approx = Q(:,1:m)*z;
    res_norm = norm(A*x_approx - b);
end



function [Q,H, lambdaKrylov, lambdaArnoldi]=arnoldi(A,b,m)
    % [Q,H]=arnoldi_m(A,b,m)
    % A simple implementation of the arnoldi_m method.
    % The algorithm will return an arnoldi_m "factorization":
    %   Q*H(1:m+1,1:m)-A*Q(:,1:m)=0
    % where Q is an orthogonal basis of the Krylov subspace
    % and H a Hessenberg matrix.
    %
    % Example:
    %  A=randn(100); b=randn(100,1);
    %  m=10;
    %  [Q,H]=arnoldi_m(A,b,m);
    %  should_be_zero1=norm(Q*H-A*Q(:,1:m))
    %  should_be_zero2=norm(Q'*Q-eye(m+1))
    n=length(b);
    Q=zeros(n,m+1);
    Q(:,1)=b/norm(b);

    t0=0;
    t1=0;
    s = 2;

    for k=1:m
        w=A*Q(:,k); % Matrix-vector product
        % with last element
        %%% Orthogonalize w against columns of Q
        % replace this with a orthogonalization
        [h,beta,worth]=repeatedGS(Q,w,k,s);
        %[h,beta,worth]=classicGS(Q,w,k);
        %[h, beta, worth] = modifiedGramSchmidt(Q, w, k);
        %%% Put Gram-Schmidt coefficients into H
        H(1:(k+1),k)=[h;beta];


        %%% normalize
        Q(:,k+1)=worth/beta;
    end
end



function [t, beta, worth] = repeatedGS(Q, w, k, s)
    t = 0;
    for i = 1:s
        h = Q(:, 1:k)'*w;
        w = w - Q(:, 1:k)*h;
        t = t + h;
    end
    worth = w;
    beta = norm(w);
end