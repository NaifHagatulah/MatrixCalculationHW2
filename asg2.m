%% conjugate gradient method for solving Ax=b

close all;n=100;
e = ones(n,1);
A = -spdiags([e -6*e e], -1:1, n, n);
b = kron(ones(n/2,1),[0;1])+ones(n,1);

it = 50;
%{
alpha = zeros(it,1);
x = zeros(size(b,1),it);
r = zeros(size(b,1),it);
beta = zeros(it, 1);
p = zeros(size(b,1), it);
z = zeros(size(b,1),it);
z_err = zeros(it,1);
cg_times = zeros(it,1);
cg_res_norm = zeros(it,1);
cg_res_norm(1) = norm(A*x(:,1) - b) / norm(b);
z_err(1) = norm(A*z(:,1) - b) / norm(b);
r(:, 1) = b;
rLSQ(:,1)
p(:, 1) = r(1);
for m = 2:it
    tic
    alpha(m) = r(:,m - 1)'*r(:,m - 1)/(p(:,m - 1)'*A*p(:,m - 1));
    x(:,m) = x(:,m - 1) + alpha(m)*p(:,m - 1);
    r(:,m) = r(:,m-1) - alpha(m)*A*p(:,m-1);
    beta(m) =  r(:,m)'*r(:,m)/(r(:,m-1)'*r(:,m-1));  
    p(:,m) = r(:,m) + beta(m)*p(:,m-1);
    cg_res_norm(m) = norm(A*x(:,m) - b) / norm(b);
    cg_times(m) = toc; 
    z(1:m,m) = (A*x(:,1:m))'*A*x(:,1:m)\(A*x(:,1:m))'*b;
    z_err(m) = norm(A*z(:,m) - b) / norm(b);
end 
%}

[x_CG, rhist_CG] = CG(A,b,it,false);
[x_CGNE, rhist_CGNE] = CG(A,b,it,true);
[x_CGLSQ, rhist_CGLSQ] = cgne(A,b,it);

rhist_CGLSQ = rhist_CGLSQ(1:end); % CGLSQ has one more iteration than CG and CGNE


[error, x_approx, res_norm, GMRES_times] = GMRES_timed(A,b,it);
%figure;
%hold on
%plot(1:it, cg_times, 'r')
%plot(1:it, GMRES_times, 'b')
%legend("CG times", "GMRES times")
figure;
hold on
semilogy(1:it, res_norm, 'r')

semilogy(1:it, rhist_CG, 'b')

semilogy(1:it, rhist_CGLSQ, 'g')
legend('GMRES', 'CG', 'CGNE')
title('Conjugate Gradient Method vs GMRES')

%% 1b)




































































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
end


function [X_res, rhist] = cgne(A, b, m)
    X_res = zeros(size(b,1),m);
    rhist = [];
    for k = 1:m
        [~, ~, x_hist] = CG_matrix(A,b,k,false);
        B = A * x_hist;
        if (det(B'*B) > 0.001)
            z = (B'*B)\(B'*b);
            X_res(:,k) = x_hist*z;
            x = x_hist * z;
            r = A*x - b;
            disp(norm(r))
            rhist = [rhist, norm(r)];
        else
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

function [x, rhist] = CG2(A,b,N,NE)
    if NE == true
        AT = A';
    else
        AT = 1;
    end
    x = zeros(size(b,1),N);
    r = AT*b;
    p = r;
    rhist = [];
    for k = 1:N-1
        rr = r'*r;
        Ap = AT*A*p;
        alpha = rr/(p'*Ap);
        x(:,k+1) = x(:,k) + alpha*p;
        r = r - alpha*Ap;
        beta = (r'*r)/rr;
        p = r + beta*p;
        rhist = [rhist, norm(r)];
    end
end

function [x, rhist] = CGLSQ2(A,b,N)
    [X, ~] = CG2(A,b,N,false);
    rhist = [];
    for k=1:size(X,2)
        AX = A*X;
        z = (AX'*AX)\(AX'*b);
        x = X*z;
        rhist = [rhist, norm(A*x - b)];
    end
end

function [x, rhist] = CGLSQ(A,b,N)
    X = [zeros(size(b))];
    z = zeros(size(X,2),1);
    r = b;
    p = r;
    rhist = [];
    for k = 1:N
        disp("it " + k)
        rr = r'*r;
        Ap = A*p;
        alpha = rr/(p'*Ap);
        x= x + alpha*p;
        X = [X, X*z];
        disp(size(X))
        r = r - alpha*Ap;
        beta = (r'*r)/rr;
        p = r + beta*p;
        rhist = [rhist, norm(r)];
    end
    x = X(:,end);
end

function [x, rhist] = CGLSQ_N(A,b,N)
    [x, rhist2] = CG(A,b,1,true);
    X = [x];
    X_it = [x];
    z = zeros(size(X,2));
    rhist = [];
    for k = 2:N
        B = A*X;
        r = B'*b;
        p = r;
        rr = r'*r;
        Bp = B'*B*p;
        alpha = rr/(p'*Bp);
        z = z + alpha*p;
        X_it = [X_it, X*z];
        r = r - alpha*Bp;
        beta = (r'*r)/rr;
        p = r + beta*p;
        rhist = [rhist, norm(r)];
        [x, rhist2] = CG(A,b,k,false);
        X = [X, x];
        z = zeros(k);
    end
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


function[error, x_approx, res_norm, times] = GMRES_timed(A,b,iterations)
    correct = A\b;
    n = length(b);
    x_approx = zeros(n, iterations);
    x_approx(:, 1) = zeros(n,1);
    r0 =  b  - A*x_approx(:,1); 
    error = zeros(iterations, 1);
    error(1) = norm(x_approx(:,1) - correct) / norm(correct);
    res_norm = zeros(iterations, 1);
    res_norm(1) = norm(A*x_approx(:,1) - b) / norm(b);
    times = zeros(iterations, 1);
    for m = 2:iterations
        tic
        e1 = [1; zeros(m,1)];
        [Q, H] = arnoldi(A,r0,m);
        z = (H\e1)*norm(b);
        x_approx(:,m) = Q(:,1:m)*z;
        error(m) = norm(x_approx(:,m) - correct) / norm(correct);
        res_norm(m) = norm(A*x_approx(:,m) - b) / norm(b);
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