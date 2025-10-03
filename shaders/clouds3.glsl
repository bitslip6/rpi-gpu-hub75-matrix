/*

    -7 chars by @iapafoto
    -2 chars by @bug + fix for div by zero
    
        thanks!!  :D
    
*/

void mainImage(out vec4 o, vec2 u) {
    float i,d,s,t = iTime;
    vec3 p = iResolution;
    u = (u-p.xy/2.)/p.y;
    for(o*=i; i++<1e2;
        d += s = .05+.2*abs(6.+p.y),
        o += 1./s + 1./length(u+u-.8))
        for (p = vec3(u * d, d + t),
             s = .01; s < 2.; s += s )
             p.yz -= abs(dot(sin(.2*t + .3*p / s ), p-p+s));
    o = tanh(o/1e3);
}

