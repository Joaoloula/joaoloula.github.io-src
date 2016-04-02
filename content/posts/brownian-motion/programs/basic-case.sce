//This program presents the Brownian movement as a limit of iterated sums of equidistributed
//discrete random variables, and extends this result to higher dimensions
function output = step()
    //Simulation of an equidistributed (in 1 dimension) step of length 1
    rand('uniform');
    if (rand(1, 1) >= 0.5) then
        output = 1;
    else
        output = -1;
    end
endfunction

function output = mouvementBrownien(quantElem, n)
    //Function that takes as inputs the number of elements
    //of the vector and a value for n, and returns a 
    //1D Brownian movement vector
    output = zeros(1, quantElem);
    for i = 2:quantElem
        output(i) = output(i-1) + step();
    end
    output = output./sqrt(n)
endfunction

n = input("What is the value of n?")
//With large value of n, we can verify empirically
//the convergence by the law of large numbers and
//the central limit theorem
t = input("What is the value of t?")
quantElem = floor(n*t);

mouvementX = mouvementBrownien(quantElem, n)';
mouvementY = mouvementBrownien(quantElem, n)';
mouvementZ = mouvementBrownien(quantElem, n)';
//Simulation of a 3-dimensional movement with independent components
temps = (1:quantElem)'./(quantElem/n);

//Different possibilities of graph outputs as proof of concept
//plot2d(mouvementX)
//plot2d(mouvementX, mouvementY);
param3d1(mouvementX, mouvementY, mouvementZ);
//param3d1([mouvementX, mouvementX, mouvementY], [mouvementY, mouvementZ, mouvementZ], list([temps, temps, temps], [color("scilab red2"), color("scilab green4"), color("scilab blue3")]));
