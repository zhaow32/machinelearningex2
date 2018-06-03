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


% h=g(X*theta),where g is the sigmoid function;
h_theta=sigmoid(X*theta);
%we only regularized parameter from theta(1)to the end but not theta(0);
reg_theta=theta(2:end,:);

%this is the regularization term: lambda/2m*sum(reg_theta^2)
reg_param=lambda/(2*m)*(reg_theta'*reg_theta);

cost=(y'*log(h_theta))+((1-y)'*log(1-h_theta));
J=-(1/m*cost)+reg_param;
% =============================================================
pd=1/m*(X'*(h_theta-y));
pd_reg_param=(lambda/m)*theta;


pd_reg_param(1) = 0;
grad = pd + pd_reg_param;
end
