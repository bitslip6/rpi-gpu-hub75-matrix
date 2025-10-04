// https://www.shadertoy.com/view/tcSfRc
#define T (iTime)

vec4 blood(vec2 u) {
    float i,a,d,s,t=.4*iTime;
    vec3  p;
    vec4 o = vec4(0);
    for(o*=i; i++<64.;
        d += s = .01 + abs(s) * .4,
        o.r+=d/s)
        for (p = vec3(u * d, d + t),
            s = min(cos(p.z), 6. - length(p.xy)),
            a = .8; a < 16.; a += a)
            p += cos(t+p.yzx)*.2,
            s += abs(dot(sin(t+.2*p.z+p * a), .6+p-p)) / a;
    return o * 2e1;
}

vec4 fire(vec2 u) {
    float i, d, s, n;
    vec3 p;
    vec4 o = vec4(0);
    for(; i++<1e2; ) {
        p = vec3(u * d, d);
        p += cos(p.z+T+p.yzx*.5)*.6;
        s = 6.-length(p.xy);
        p.xy *= mat2(cos(.3*T+vec4(0,33,11,0)));
        for (n = 1.6; n < 32.; n += n )
            s -= abs(dot(sin( p.z + T + p*n ), vec3(1.12))) / n;
        d += s = .01 + abs(s)*.1;
        o += 1. / s;
    }
    return (vec4(5,2,1,1) * o * o / d);
}

void mainImage(out vec4 o, in vec2 u) {
    float s=.1,d=0.,i=0.;
    vec3  p = iResolution;
    u = (u-p.xy/2.)/p.y;

    o = mix(fire(u), blood(u), .9);
    o = tanh(o  / 5e5 );

}
