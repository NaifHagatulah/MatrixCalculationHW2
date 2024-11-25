load('cgn_illustration.mat')

compare_gmres_cgn(A, b)

% Main script to compare GMRES and CGN
function compare_gmres_cgn(A, b)
    max_iter = 10;  % Maximum iterations
    
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
        [error, x_gmres, res_norm] = GMRES(A, b, k);
        gmres_times(k) = gmres_times(k-1) + toc(gmres_start);
        gmres_residuals(k) = norm(A*x_gmres - b);
        
        % CGN
        cgn_start = tic;
        [x_cgn, r_hist] = CG(A, b, k, true);
        cgn_times(k) =  toc(cgn_start);
        cgn_residuals(k) = norm(A*x_cgn - b);
    end
    disp(cgn_residuals(max_iter))
    disp(r_hist(max_iter))
    % Plot results
    figure(1)
    semilogy(0:max_iter-1, gmres_residuals, 'b-', 'LineWidth', 1.5)
    hold on
    semilogy(0:max_iter-1, cgn_residuals, 'r--', 'LineWidth', 1.5)
    xlabel('Iteration')
    ylabel('||Ax - b||_2')
    legend('GMRES', 'CGN')
    grid on
    hold off
    
    figure(2)
    semilogy(gmres_times, gmres_residuals, 'b-', 'LineWidth', 1.5)
    hold on
    semilogy(cgn_times, cgn_residuals, 'r--', 'LineWidth', 1.5)
    xlabel('CPU-time (seconds)')
    ylabel('||Ax - b||_2')
    legend('GMRES', 'CGN')
    grid on
    hold off
end




function [x, rhist] = CG(A,b,N,NE)
    if NE == true
        AT = A';
    else
        AT = 1;
    end
    x = zeros(size(b));
    r = AT*b;
    p = r;
    rhist = [];
    for k = 1:N
        rr = r'*r;
        Ap = AT*A*p;
        alpha = rr/(p'*Ap);
        x = x + alpha*p;
        r = r - alpha*Ap;
        beta = (r'*r)/rr;
        p = r + beta*p;
        rhist = [rhist, norm(r)];
    end
end


function[error, x_approx, res_norm] = GMRES(A,b,iterations)
    correct = A\b;
    n = length(b);
    x_approx = zeros(n, iterations);
    x_approx(:, 1) = zeros(n,1);
    r0 =  b  - A*x_approx(:,1); 
    error = zeros(iterations, 1);
    error(1) = norm(x_approx(:,1) - correct) / norm(correct);
    res_norm = zeros(iterations, 1);
    res_norm(1) = norm(A*x_approx(:,1) - b) / norm(b);
    
    for m = 2:iterations

        e1 = [1; zeros(m,1)];
        [Q, H] = arnoldi(A,r0,m);
        z = (H\e1)*norm(b);
        x_approx(:,m) = Q(:,1:m)*z;
        error(m) = norm(x_approx(:,m) - correct) / norm(correct);
        res_norm(m) = norm(A*x_approx(:,m) - b) / norm(b);
    end

end

function[x_approx] = GMRES_opt(A,b, m)
    x_approx = zeros(length(b), 1);
    r0 =  b  - A*x_approx;
    [Q, H] = arnoldi(A,r0,m);     
    e1 = [1; zeros(m,1)];  
    z = (H\e1)*norm(b);
    x_approx = Q(:,1:m)*z;
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