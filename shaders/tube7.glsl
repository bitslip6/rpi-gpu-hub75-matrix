#define r(a) mat2(cos(a - vec4(0,11,33,0)))


void mainImage(out vec4 o,in vec2 F)
{
    vec2 R=iResolution.xy;

    vec3 P,Q;
    float i=0.,g=0.,d=.1,a,h=.4;

    o=vec4(0.);

    for(;i<99.&&d>1e-4;i++)
    {
        // Advance ray
        g+=d*.3;

        // Position along ray
        P=vec3((F-.5*R)/R.y*g,g);

        P.z+=iTime;
        P.xy *= r(P.z*0.8);
        // Base distance field
        d=1.-abs(P.y);//(length(P.xy)-1.);

        // Repeated structure
        float s=1.;

        for(a=2.;a<6e2;a+=a)
        {
            Q=P*a;

            Q.z+=d*12.5;

            d-=abs(dot(sin(Q),Q-Q+1.))/(a*3.);
        }

        // Fabrice-style rendering
        //o+=1./(.01+abs(d)*.8);
        //o += vec4(1.,.3,.1,1.)/(.01+abs(d)*.8);
        //o += (.5+.5*cos(P.z+vec4(0,2,4,0)))/(.01+abs(d)*.8);
        o += (.5+.5*cos(d*40.+P.z*2.+vec4(1.,4.4,4,0)))/(.006+abs(d)*.08);
    }

    // Tone mapping
    o=tanh(o/2e4/length((F-.5*R)/R.y));
}
