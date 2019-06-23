A = [ 0.4873 -0.8732;
      0.6072  0.7946;
      0.9880 -0.1546;
     -0.2142 -0.9768;
     -0.9871 -0.1601;
      0.9124 0.4093];
 b = [1; 1; 1; 1; 1; 1];
 V = [0.1562  0.9127 1.0338 0.8086 -1.3895 -0.8782  0.1562;
     -1.0580 -0.6358 0.1386 0.6406  2.3203 -0.8311 -1.0580];
 t = 0.1;
 c = [1; -1];
 x0 = [0; 0];
 k = 0;
 m = 6;
 gamma = 1.5;
 epsilon = 10^-3;
 alfa = 0.5;
 beta = 0.8;
 xk = x0;
 centr = [];
 s = 0
 while(m/t > epsilon)
     s = s+1;
     xk = getNextX(xk, A, b, t, c, alfa, beta)
     centr = [centr, xk];
     t = gamma * t;
     k = k + 1;
 end
 xk
 linx = linprog(c, A, b);
 x1 = -2.5:0.01:2;
 x2 = -1.5:0.01:3.0;
 [X1, X2] = meshgrid(x1, x2);
 fill(V(1,:), V(2,:), 'r');
 grid on
 hold on
 scatter(centr(1,:), centr(2,:), 10, 'filled');
 scatter(linx(1), linx(2), 15, 'filled');
 
 Z = arrayfun(@fbar, X1, X2);
 J = arrayfun(@flin, X1, X2);
 v = [-0.6, 0.12, 1.24];
 contour(X1,X2,Z,v, 'black', 'ShowText', 'on');
 contour(X1,X2,J, '--black')
 
 function y = flin(x1, x2)
    c = [1 -1];
    y = c * [x1; x2];
 end
 
 function y = fbar(x1, x2)
 A = [ 0.4873 -0.8732;
      0.6072  0.7946;
      0.9880 -0.1546;
     -0.2142 -0.9768;
     -0.9871 -0.1601;
      0.9124 0.4093];
 b = [1; 1; 1; 1; 1; 1];
    if sum(b - A * [x1; x2] < 0) == 0
       y = -sum(log(b - A * [x1; x2]));
    else
       y = NaN;
    end
 end
 
function nextX = getNextX(currentX, A, b, t, c, alfa, beta)
    nextX = currentX - getS(currentX, A, b, t, c, alfa, beta) * gradientKW(currentX, A, b, t, c)^-1 * gradient(currentX, A, b, t, c);
end
 
function grad = gradient(x, A, b, t, c)
    grad = t * c + (A')*(1./(b-A*x));
end

function grad = gradientKW(x, A, b, t, c)
    grad = ((diag(1./(b-A*x))*A)')*(diag(1./(b-A*x))*A);
end

function f = func(x, A, b, t, c)
    f = t * c' * x - sum(log(b-A*x));
end

function sk = getS(x, A, b, t, c, alfa, beta)
    grad = gradient(x, A, b, t, c);
    gradkw = gradientKW(x, A, b, t, c);
    s = 1;
    while(func(x - s * gradkw^-1 * grad, A, b, t, c) >= func(x, A, b, t, c) - s * alfa * transpose(grad) * gradkw^-1 * grad)
        s = s * beta;
    end
    sk = s;
end