//We are now interested in brownian movement in continuous time. To simulate this
//phenomenum, we send the time step in the previous programs to zero and then analyse the
//behavior of velocity and position in big timescales as realizations of random variables

//The user chooses the timespan
a = input("What is the value of t?");
//For better visualization, we consider zero initial position and velocity
//Evidently, for the laws considered in this program, it is just a matter of 
x = 0;
v = 0;
//Lambda represents the relaxation time and D the diffusivity
//The values used here represent the ratio of these two parameters
//for the case of hemoglobin in the blood
lambda=100000000;
D=1;
function output = grandx(t)
    //We simulate the behavior of X(t) in the approximation of vanishingly small time steps
    position = zeros(1, t);
    position(1)=x;
    for i=2:t;
        Y=grand(1,1,'nor',0,lamda*D*(1-exp(-1*lambda)));
        position(i)=position(i-1)+Y
    end
output=position(t);
endfunction
//We create a vector with many simulations of position values
xdata=zeros(1,100);
for i=1:100;
    xdata(i)=grandx(a);
end
//We introduce the coordinate vector that we`ll use for the histogram
//It is the linspace of the width of our data, increased by a small quantity
//to avoid a known bug in scilab

x = linspace(min(xdata)-%eps, max(xdata)+%eps, n+1);
//We finally plot the histograms to evaluate the tendencies in the data
histplot(x, xdata);
