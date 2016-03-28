function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
t = theta(2:end);

H = X * theta;
J = (sum((H - y).^ 2))/(2*m) + ((lambda/(2*m)) * sum(t .^ 2));

grad(1) = (sum((H - y) .* X(:,1)))/m;

for j =2:(size(theta))
  grad(j) = (sum((H - y) .* X(:,j))/m + ((lambda/m)*theta(j))); % falta ajustar para que se sumen a la vez todo los theta_j
end;







% =========================================================================

grad = grad(:);

end
