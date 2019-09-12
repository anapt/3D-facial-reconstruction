fileID = fopen('./happiness/ground_truth/happy.txt','r');
A = fscanf(fileID,'%f');
happy = A;

fileID = fopen('./neutral/ground_truth/neutral.txt','r');
A = fscanf(fileID,'%f');
neutral = A;

fileID = fopen('./sadness/ground_truth/sad.txt','r');
A = fscanf(fileID,'%f');
sad = A;

fileID = fopen('./surprise/ground_truth/surprised.txt','r');
A = fscanf(fileID,'%f');
surprised = A;

figure(1)
y = [1:64];
scatter(y, happy , 'blue');
hold on
scatter(y, sad, 'red');
scatter(y, neutral, 'green')
scatter(y, surprised, 'black');

hold off

figure(2)
fileID = fopen('./surprise/ground_truth/left_limit.txt','r');
A = fscanf(fileID,'%f');
less_happy = A;

fileID = fopen('./surprise/ground_truth/right_limit.txt','r');
A = fscanf(fileID,'%f');
more_happy = A;

scatter(y, surprised, 'blue')
hold on
scatter(y, less_happy, 'red')
scatter(y, more_happy, 'green')
legend('surprised','less surprised', 'more surprised')