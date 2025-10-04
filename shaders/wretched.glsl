// https://www.shadertoy.com/view/tXsyR8

vec4 fire(vec3 p) {
    float s, i, d, n, T = iTime;
    vec4 o = vec4(0);
    p += cos(p.z+T+p.yzx*.5)*.6;
    s = 6.-length(p.xy);
    p.xy *= mat2(cos(.3*T+vec4(0,33,11,0)));
    for (n = 1.6; n < 32.; n += n )
        s -= abs(dot(sin( p.z + T + p*n ), vec3(1.12))) / n;
    s = (.01 + abs(s)*.15);
    o += 1. / s;
    return (vec4(7,2,1,1) * o * o );
}

void mainImage(out vec4 o, vec2 u) {
    float d=4.*texelFetch(iChannel0, ivec2(u)%1024, 0).a,
          i,s,w,l, T = iTime*.3;
    vec3  q,p = iResolution;
    u = ( u - p.xy/2. ) / p.y;
    for(o*=i; i++ < 64.; o += d / s + 4.*fire(q)) {
        q = p = vec3( u*d, d + T * 4.),
        p.xy *= mat2(cos(.001*T+p.z*.1+vec4(0,33,11,0)));
        p *= .3;
        w = .25;
        p.x-=1.5;
        p += cos(T+p.yzx);
        for (int i; i++ < 7; w *= l )
            p *= l = 1.25/dot( p = abs(sin(p))-1. , p);
        d += s = .002+.5*abs(length(p)/w) ;
    } 
    o = tanh(o/2e5);
}

