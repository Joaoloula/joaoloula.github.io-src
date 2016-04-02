//In this program, new parameters are introduced to account for the physical characteristics
//of the system: we imagine a small particule imersed in a fluid that suffers a large number of
//independent identically distributed collisions modulated by a small time step, as well as a 
//dampening factor introduced by the fluid.
x = 0;
v = 1;
//Initial values for position and speed

n = input("What is the value of n?");
a = input("What is the value of a?");
b = input("What is the value of b?");
alpha = input("What is the value of alpha?");
//Here, "alpha" represents the size of a time step, "a" is a dampening coefficient (a<1: dampening,
//a>1:acceleration), "b" is the noise on the velocity (it functions as the standard deviation
//of a centered gaussian) and "n" is the length of the vector
temps = [0: alpha : (n-1)*alpha]
vitesse = zeros(1, n);
vitesse(1)=v;
//We build the velocity vector through a law of recursion over v
for i=2:n;
    Y=grand(1,1,'nor',0,1);
    vitesse(i)=(vitesse(i-1)*a)+b*Y;
end
position = zeros(1, n);
//We do the same for the position, now using the values found for V
for i=2:n;
    position(i)=(position(i-1))+alpha*vitesse(i-1);
end

//Second (X, V) couple, can be added to simulate, for example, Brown's pollen experiment
//vitesse2 = zeros(1, n);
//vitesse2(1)=v;
//for i=2:n;
//    Y=grand(1,1,'nor',0,1);
//    vitesse2(i)=(vitesse2(i-1)*a)+b*Y;
//end
//position2 = zeros(1, n);
//for i=2:n;
//    position2(i)=(position2(i-1))+alpha*vitesse2(i-1);
//end
//Affichage des graphes
plot3d(position);
param3d1(position, vitesse, temps);
//param3d1(position, position2, temps);
//plot2d(position, position2)
