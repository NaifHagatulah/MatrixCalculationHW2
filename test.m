clear; close all; format long;

n=100; rand('state',5);
alpha = [1, 5, 10, 100];


%% 1a)
iterations = 30;
for i = 1:4
    A = sprand(n,n,0.5);
    A = A + alpha(i)*speye(n); A=A/norm(A,1);
    b = rand(n,1);
    [error, x_approx, res_norm] = GMRES(A, b, iterations);
    figure;
    semilogy(1:iterations, error)
    title("Convergence for alpha = " + alpha(i))
    hold on;
    xlabel("Iterations")
    ylabel("Magnitude")
    ylim([0, 1.2])
    semilogy(1:iterations, res_norm)
    legend('Relative Error', 'Relative Residual Norm')
    %saveas(gcf, "ConvergenceGMRESvsBackslashAlpha" + alpha(i) + ".png")
end


%% 1b)


for i = 1:4
    % Generate matrix A and normalize
    A = sprand(n, n, 0.5);
    A = A + alpha(i) * speye(n);
    A = A / norm(A, 1);
    
    % Compute eigenvalues
    [V, lambda] = eig(full(A));
    eigenvalues = diag(lambda);
    
    % Compute circle parameters
    [radius, center] = circle(A);
    
    
    % Plot eigenvalues and enclosing circle
    figure;
    plot(real(eigenvalues), imag(eigenvalues), '*'); % Eigenvalues
    hold on;
    theta = linspace(0, 2 * pi, 100);
    circle_x = real(center) + radius * cos(theta);
    circle_y = imag(center) + radius * sin(theta);
    plot(circle_x, circle_y, 'r-'); % Enclosing circle
    axis equal; % Keep aspect ratio equal for accurate visualization
    title(['Eigenvalues for \alpha = ', num2str(alpha(i))]);
    legend('Eigenvalues', 'Enclosing Circle');
    xlabel("Real")
    ylabel("Imaginary")
    hold off;
    %saveas(gcf, "EigenvaluesAlpha" + alpha(i) + ".png")
end




%%1c)

% Parameters
alphas = [1, 100];
n_values = [200, 500, 1000];
iterations = [5, 10, 20, 50, 100];

% Storage for results
gmres_results = struct();
backslash_results = struct();

for alpha = alphas
    for n = n_values
        % Generate matrix A and vector b
        A = sprand(n, n, 0.5);
        A = A + alpha * speye(n);
        A = A / norm(A, 1); % Normalize A
        b = rand(n, 1);
        
        % GMRES timing and residual norm
        for m = iterations
            tic;
            x_approx = GMRES_opt(A, b, m); 
            gmres_time = toc;

            
            % Store GMRES results
            gmres_results(alpha, n, m).resnorm = norm(A*x_approx - b) / norm(b);
            gmres_results(alpha, n, m).time = gmres_time;
        end
        
        % Backslash operator timing and residual norm
        tic;
        x_exact = A \ b;
        backslash_time = toc;
        
        % Store backslash results
        backslash_results(alpha, n).resnorm = norm(A * x_exact - b);
        backslash_results(alpha, n).time = backslash_time;
    end
end

% Display results

for alpha = alphas
    for n = n_values
        fprintf('Results for alpha = %d, n = %d\n', alpha, n);
        fprintf('m\tGMRES Residual Norm\tGMRES Time\tBackslash Residual Norm\tBackslash Time\n');
        for m = iterations
            fprintf('%d\t%.4e\t%.4e\t%.4e\t%.4e\n', m, gmres_results(alpha, n, m).resnorm, gmres_results(alpha, n, m).time, backslash_results(alpha, n).resnorm, backslash_results(alpha, n).time);
        end
        fprintf('\n');
    end
end




































function [radius, center] = circle(A)
    [V, D] = eig(full(A));
    lambda = diag(D);
    center = mean(lambda);
    radius = max(abs(lambda - center));
end





function [error, x_approx, res_norm] = GMRES(A,b,iterations)
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
        
        
function [h, beta, worth] = classicGS(Q, w, k)
    %h = Q(1:k,:)'*w;
    h = Q(:,1:k)' * w; % Project w onto the first k columns of Q
    worth = w - Q(:,1:k)*h;
    beta = norm(worth);
end