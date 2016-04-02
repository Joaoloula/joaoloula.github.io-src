//We now introduce a force that penalizes the velocity proportionally to how far away from the center
//the particle is. The consequence (as can be seen by looking at the eigenvalues of the recursion law
//matrix) is a stabilizing system that's enveloped by an exponential, a classic case of dampening.
//Though the recursion laws found are not pretty, it is interesting to see that the results have a
//fruitful physical interpretation (the force introduced can be thought of as a planet's gravitational 
//attraction, a charged electron etc.)

//The user chooses the values of t and K
t = input("What is the value of t?");
K = input("What is the value of K?");
//The choice of initial position and velocity are again arbitrary
//We are, however, tempted to choose values far from the origin in
//order to observe a stabilizing behaviour
x = 100;
v =100;
//This choice of lambda and D was made merely for pleasantness of scale
lambda=1;
D=1;
//The value of beta will influence the intensity of the force that pulls the particle towards the center
xbeta=0.9;
//We initialize the position and velocity vectors
vitesse = zeros(1, t*K);
vitesse(1)=v;
position = zeros(1, t*K);
position(1) = x;
alpha = 1/K;
for i=2:t*K;
    //We now treat both variables in the same loop: we have no other choice given the intricacy of
    //their bond in this situation. We'll make use of a law of recursion determined by the dynamical
    //system itself
    Y=grand(1,1,'nor',0,1);
    vitesse(i)=(vitesse(i-1)*(1-(lambda*alpha)))+(sqrt(lambda*lambda*D*alpha)*Y)-xbeta*position(i-1);
    position(i)=position(i-1)+(vitesse(i-1)*alpha);
end
//We finally plot the results
plot2d(position);
