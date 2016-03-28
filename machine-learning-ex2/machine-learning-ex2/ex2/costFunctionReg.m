function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(X * theta);
S = -y.*log(H) - (1 - y).*log(1-H);
theta_sq =(theta(2:columns(X),:) .^ 2);
J = sum(S)/m + (lambda/(2*m))*sum(theta_sq);

for iter = 1:columns(X)
    S_2 = (sum(X(:,iter)' *(H - y)))/m;
    if(iter != 1)
      S_2 = S_2 + (lambda/m) * theta(iter,:);
     end
     
    grad(iter) = S_2;
end


% =============================================================

end
