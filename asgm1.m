clear; close all; format long;

n=100; rand('state',5);
A = sprand(n,n,0.5);

%% 1a)
iterations = 30;
for alpha = [1, 5, 10, 100]
    A = A + alpha*speye(n); A=A/norm(A,1);
    b = rand(n,1);
    [error, x_approx] = GMRES(A, b, iterations);
    figure;
    semilogy(1:iterations, error)
    title(alpha)
end




















































function[error, x_approx] = GMRES(A,b,iterations)
    correct = A\b;
    n = length(b);
    x_approx = zeros(n, iterations);
    x_approx(:, 1) = zeros(n,1);
    r0 =  b  - A*x_approx(:,1); 
    error = zeros(iterations, 1);
    error(1) = norm(x_approx(:,1) - correct) / norm(correct);
    
    for m = 2:iterations

        e1 = [1; zeros(m,1)];
        [Q, H] = arnoldi(A,r0,m);
        z = (H\e1)*norm(b);
        x_approx(:,m) = Q(:,1:m)*z;
        error(m) = norm(x_approx(:,m) - correct) / norm(correct);
    end

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
        
        
function [h, beta, worth] = classicGS(Q, w, k)
    %h = Q(1:k,:)'*w;
    h = Q(:,1:k)' * w; % Project w onto the first k columns of Q
    worth = w - Q(:,1:k)*h;
    beta = norm(worth);
end
    