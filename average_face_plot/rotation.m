function rot_mat_so3 = rotation(a,b,c)
a = 10 * a;
b = 10 * b;
c = 10 * c;

c1 = cosd(c);
c2 = cosd(b);
c3 = cosd(a);

s1 = sind(c);
s2 = sind(b);
s3 = sind(a);

rot_mat_so3 = ([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 - c1*c3*s2],
                    [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3],
                   [-s2, c2*s3, c2*c3]]);


end

