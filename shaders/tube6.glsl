
#define T (sin(iTime*.6)*16.+iTime*1e2)
#define P(z) (vec3(cos((z)*.011)*16.+cos((z) * .012)  *24., \
                   cos((z)*.01)*4., (z)))
#define R(a) mat2(cos(a+vec4(0,33,11,0)))
#define N normalize

float boxen(vec3 p) {
    
    p = abs(fract(p/2e1)*2e1 - 1e1) - 1.;
    return min(p.x, min(p.y, p.z));

}

vec4 lights;
float map(vec3 p) {
    vec3 q = P(p.z);
    float m, g = q.y-p.y + 6.;

    m = boxen(p);
               
    p.xy -= q.xy;
    
    // squiggly line along z
    float red,blue;
    float e = min(red=length(p.xy -  sin(p.z / 12. + vec2(0, 1.3))*12.) - 1.,
                  blue=length(p.xy -  sin(p.z / 16. + vec2(0, .7))*16.) - 2.);  

    lights += vec4(1e1,2,1,0)/(.1+abs(red));
    lights += vec4(1,2,1e1,0)/(.1+abs(blue)/1e1);;

    p = abs(p);
    
    float tex = abs(length(sin(p*cos(p.yzx/3e1)*4.)/(p*4.)));     
    float tun = min(32.-p.x - p.y, 24.-p.y);


    float d = max(min(m, g), tun)-tex;
    return min(e, d);
}

void mainImage(out vec4 o, in vec2 u) {
    float i,s,d;
    vec3  r = iResolution;
    
    u = (u-r.xy/2.)/r.y;
    
    u.y -=.2;    
    o = vec4(0);
    vec3  p = P(T),ro=p,
          Z = N( P(T+2.) - p),
          X = N(vec3(Z.z,0,-Z)),
          D = N(vec3(R(sin(T*.005)*.4)*u, 1) 
             * mat3(-X, cross(X, Z), Z));
    
    for(; i++ < 1e2;)
        p = ro + D * d,
        d += s = map(p)*.8,
        o += lights + 1./max(s, .01);


    // normal
    // tetrahedron technique: https://iquilezles.org/articles/normalsSDF/
    const float h = 0.005;
    const vec2 k = vec2(1,-1);
    vec3 n = N(k.xyy*map( p + k.xyy*h ) + 
               k.yyx*map( p + k.yyx*h ) + 
               k.yxy*map( p + k.yxy*h ) + 
               k.xxx*map( p + k.xxx*h ) );

    // diffuse
    o *= (.1 + max(dot(n, -D), 0.));
    
    // reflection march
    vec4 ref;
    lights = vec4(0);
    for(p += n*.05, D = reflect(D, n), s=i=0.; i++<5e1; )
        p += D*s,
        s = map(p)*.8,
        ref +=  lights + 1./max(s, .01);

    o += o*ref;
    o = tanh(o / 1e9 * exp(vec4(1e1,2,1,0)*d/5e2));
}
