%% conjugate gradient method for solving Ax=b

close all;n=100;
e = ones(n,1);
A = -spdiags([e -6*e e], -1:1, n, n);
b = kron(ones(n/2,1),[0;1])+ones(n,1);

it = 30;

[x_CG, rhist_CG] = CG(A,b,it,false);
[x_CGLSQ, rhist_CGLSQ] = cgne(A,b,it);
[x, rhist, res_norm_lsq] = CGLSQ(A,b,it);

rhist_CGLSQ = rhist_CGLSQ(1:end); % CGLSQ has one more iteration than CG and CGNE


[x_approx, res_norm, GMRES_times] = GMRES_timed(A,b,it);
figure;
hold on
plot(1:it, res_norm, 'r')

plot(1:it, rhist_CG, 'b')

plot(1:21, res_norm_lsq(1:21), 'g')
set(gca, 'YScale', 'log')
ylabel("Residual")
xlabel("Iteration")
legend('GMRES', 'CG', 'CGLSQ')
title('CG vs GMRES vs CGLSQ')
saveas(gcf, "CGvsGMRESvsCGLSQ.png")


function [x, rhist] = CG(A,b,N,NE)
    if NE == true
        AT = A';
    else
        AT = 1;
    end
    x = zeros(size(b));
    r = AT*b; %AT = 1 if we have a symetric matrix
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
    disp("CG " + rhist(1))
end


function [X_res, rhist] = cgne(A, b, m)
    X_res = zeros(size(b,1),m);
    rhist = [];
    for k = 1:m
        [~, ~, x_hist] = CG_matrix(A,b,k,false);
        B = A * x_hist;
        if k==5
            %disp(x_hist)
        end
        if (det(B'*B) > 0.001)
            z = (B'*B)\(B'*b);
            X_res(:,k) = x_hist*z;
            x = x_hist * z;
            r = A*x - b;
            rhist = [rhist, norm(r)];
        else
            %disp("singular it " + k)
            X_res(:,k) = X_res(:,k-1);
            rhist = [rhist, rhist(k-1)];
        end
    end
end


function [x, rhist, x_hist] = CG_matrix(A,b,N,NE)
    if NE == true
        AT = A';
    else
        AT = 1;
    end
    x = zeros(size(b));
    x_hist = zeros(size(b,1),N);
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
        x_hist(:,k) = x;
    end
end

function [x, rhist, res_norm] = CGLSQ(A,b,N)
    X = [];
    XLSQ = []; 
    x = zeros(size(b));
    r = b;
    p = r;
    rhist = [];
    for k = 1:N
        rr = r'*r;
        Ap = A*p;
        alpha = rr/(p'*Ap);
        x = x + alpha*p;
        X = [X,x];
        B = A*X;
        z = (B'*B)\B'*b;
        XLSQ = [XLSQ, X*z];
        r = r - alpha*Ap;
        beta = (r'*r)/rr;
        p = r + beta*p;
        rhist = [rhist, norm(r)];

    end
    res_norm = vecnorm(A*XLSQ - b);
    disp("CGLSQ " + res_norm(1))
    x = X(:,end);
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


function[x_approx, res_norm, times] = GMRES_timed(A,b,iterations)
    n = length(b);
    x_approx = zeros(n, iterations);
    x_approx(:, 1) = zeros(n,1);
    r0 =  b  - A*x_approx(:,1); 
    res_norm = zeros(iterations, 1);
    res_norm(1) = norm(A*x_approx(:,1) - b);
    times = zeros(iterations, 1);
    for m = 2:iterations
        tic
        e1 = [1; zeros(m,1)];
        [Q, H] = arnoldi(A,r0,m);
        z = (H\e1)*norm(b);
        x_approx(:,m) = Q(:,1:m)*z;
        %error(m) = norm(x_approx(:,m) - correct) / norm(correct);
        res_norm(m) = norm(A*x_approx(:,m) - b);
        times(m) = toc;
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