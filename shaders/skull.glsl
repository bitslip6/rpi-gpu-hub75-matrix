
// "Dull Skull"
// Prologue
// 2024
// by KΛTUR
// License - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Unported License.

/*
=======================================================================

In this shader, I revisit a key element from my previous release:
"DULL SKULL"
https://www.shadertoy.com/view/DlyyWR

Initially, I created this skull, which became the foundation for the
rest of the skeleton.Building on that foundation, I explore the
intricacies of the human skeleton through simple mathematics,
focusing on the skull as a SDF mostly made of classic linear algebra
and simple Boolean operations.

=======================================================================
*/

#define MAX_STEPS 64
#define MAX_DIST 32.
#define SURF_DIST .001

#define PI 3.141592

// https://mercury.sexy/hg_sdf/
float pModPolar(inout vec2 p, float repetitions) {
    
    float angle = 2.*PI/repetitions,
          a = atan(p.y, p.x)+angle,
          r = length(p),
          c = floor(a / angle);
    a = mod(a, angle) - angle / 2.0;
    p = vec2(cos(a), sin(a)) *r ;
    if (abs(c) >= (repetitions / 2.0)) c = abs(c);
    return c;
}

//===================================================================//
// below by https://iquilezles.org/articles/distfunctions/

mat2 Rot(float a) {
    float s=sin(a), c=cos(a);
    return mat2(c, -s, s, c);
}

float sMin( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); }

float sMax( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); }

float Sphere(vec3 p,float s){
    return length(p)-s;
}

float Ellipsoid( vec3 p, vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float rBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float Capsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

float HollowSphere( vec3 p, float r, float h, float t )
{
  float w = sqrt(r*r-h*h);
  vec2 q = vec2( length(p.xz), p.y );
  return ((h*q.x<w*q.y) ? length(q-vec2(w,h)) : 
                          abs(length(q)-r) ) - t;
}

// above by https://iquilezles.org/articles/distfunctions/
//===================================================================//

vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 
        f = normalize(l-p),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f,r),
        c = f*z,
        i = c + uv.x*r + uv.y*u;
    return normalize(i);
}

float t(float time){
    float t = 3.+time*.5;
    return t += sin(time*.5)*.6;
}

vec3 Transform(vec3 p, float t){
    p.y -= .4;
    p.y += sin(t+1.6)*.3;
    p.z += sin(t*.9-1.6)*.6-.3;
    p.yz *= Rot(sin(-t+1.)*.3);
    p.xy *= Rot(cos(-t*.7+4.)*.4);
    p.xz *= Rot(sin(t*.5)*1.*cos(t*.3+1.));
    
    return p;
}

vec2 map(vec3 p){
       
    float t = t(iTime);
    mat2 ani = mat2(0);
    ani = Rot(sin(t-1.7)*.2-.1);   
    vec3 p_skull = p;
    p_skull = Transform(p,t);

    // HEAD
    vec3 p_head = p_skull;
        float d = Ellipsoid(p_head,vec3(.9,1.1,1.2));//head ellispoid
        float p_cutb = p_head.y+.7 + sin(p_head.x + sin(cos(p_head.z*1.4)) * 21.)*.02; //bottom cut plane
        p_cutb = sMin(p_cutb, Ellipsoid(p_head-vec3(0,-.3,-.2),vec3(.7)),.05);//head hole
        p_cutb = sMin(p_cutb, Ellipsoid(p_head-vec3(0,-.24,.5),vec3(.51)),.1);//head hole front
    d = sMax(p_cutb, d,.05); //bottom cut
        float p_cutf = -p_head.z+1.1; //forehead plane
    d = sMax(p_cutf, d,.2); //forehead cut
        float cuts_temple = Capsule(vec3(-abs(p_head.x),p_head.yz), vec3(-1.,-1,.8), vec3(-1.8,3,.0), .5 );//temple deepenings
    d = sMax(cuts_temple, d,.3); //temple cuts
        float bcut_temple = Capsule(p_head, vec3(-2.,-1.1,.6), vec3(2,-1.1,.6), .6 );//side cuts temple <-> head
    d = sMax(bcut_temple, d,.3); //side cuts    
    
    // UPPER JAW
    vec3 p_jaw = p_skull-vec3(0,.36,.1);
        p_jaw.yz *= Rot(PI);
        p_jaw.y -= sin(p_jaw.x*37.)*.007 - cos(p_jaw.z*59.)*.001;//distortion
        p_jaw.z *= .9;
        float ujaw = HollowSphere(p_jaw+vec3(0,-.95,.5),.38,.02,.05 );//jaw sdf
        float p_cutB = p_skull.z-.6;//cutting plane back
        ujaw = sMax(p_cutB, ujaw,.05); //jaw back cut
    vec3 p_jawsc = vec3(abs(p_skull.x),p_skull.yz); //new point def for side cuts
        p_jawsc.xy *= Rot(-1.);
        p_jawsc.yz *= Rot(-.4);
        p_jawsc.y += .3; 
        ujaw = sMax(p_jawsc.y, ujaw,.04); //side cuts
    
    d = sMin(ujaw, d,.1);//upper jaw surrounding and d     
    d -= sin(10.*p_skull.x)*sin(8.*p_skull.y)*sin(6.*p_skull.z)*.03;//distortion      
        
    // CHEEKS         
        // CHEEKBONES
    vec3 p_eyesur = p_skull-vec3(0,.3,0);
        float eyesur = Ellipsoid(vec3(abs(p_eyesur.x),p_eyesur.yz)+vec3(-.34,.5,-.87),vec3(.25,.3,.2));//cheekbones   
        eyesur += sin(12.*p_skull.x)*sin(11.*p_skull.y)*sin(13.*p_skull.z)*.02;//distortion
    d = sMin(eyesur, d,.1);//union cheekbones and d
    
        // ZYGOMATIC ARCH
    vec3 p_zyg = vec3(abs(p_skull.x),p_skull.yz);
        p_zyg.x += sin(p_zyg.z*4.+PI)*.08;
        p_zyg.y += cos(p_zyg.z*9.)*.03;
        float zyg = Capsule(p_zyg,vec3(.5,-.3,.8),vec3(.75,-.3,0.1),(p_zyg.z)*.1);
    d = sMin(d,zyg,.06);
   
    // NOSE
        // NOSE BONE
    vec3 p_nbone = p_skull;
        p_nbone.yz *= Rot(-2.2);
        float nbone = HollowSphere(p_nbone+vec3(0,-1.,.4),.1,0.08,.04 );
    d = sMin(d,nbone,.05);
    
        // NOSE HOLE
    vec3 p_nose = p_skull;//vec3(abs(p_skull.x),p_skull.yz);
        p_nose.xy *= Rot(.25);
        float nose = Ellipsoid(p_nose-vec3(.04,-.35,1.),vec3(.03,.1,.8));
        p_nose.xy *= Rot(-.4);
        nose = sMin(nose,Ellipsoid( p_nose-vec3(.02,-.36,1.),vec3(.04,.1,.8)),.1);
    vec3 p_bone = p_skull;
            
    d = sMax(nose, d,.06); //nose subtraction
    d = sMax(Ellipsoid(p_nose+vec3(0.,.3,-.4),vec3(.1,.1,.6)), d,.1); //nose subtraction
                
    // LOWER JAW
    vec3 pN = p_skull;//null - parent rotation
        pN.z -= .5;
        pN.y += .4;
        pN.yz *= ani;
        pN.z += .5;
        pN.y -= .4;
        pN -= sin(pN.y*15.)*.01 - cos(pN.z*39.)*.002;//distortion
        
        //CHIN
    vec3 p_ljaw = pN;
        p_ljaw.y *= .8;
        p_ljaw.z -= sin(pN.y*26.)*.008;
        p_ljaw.y -= cos(pN.x*15.+sin(pN.y*7.)*2.)*.01;
        float ljaw = HollowSphere(p_ljaw+vec3(0,.77,-.74),.38,0.03,.04 );//chin  
        ljaw = sMax(p_ljaw.z-.65,ljaw,.1);
        
        // MANDIBLE BOTTOM & BACK
    vec3 p_maB = vec3(abs(pN.x),pN.yz);
        p_maB.yz *= Rot(-1.3);
        p_maB.xz *= Rot(-.34);
        p_maB.xy *= Rot(-.39);
        p_maB -= vec3(0.85,.0,.63);
        ljaw = sMin(ljaw,rBox(p_maB,vec3(0.,smoothstep(0.,6.,abs(-p_maB.z)+.9),.45),.04),.17);
        ljaw = sMax(Ellipsoid(p_maB-vec3(.0,.0,-.55),vec3(.5,.15,.26)),ljaw,.04);
        p_ljaw -= sin(p_ljaw.y*22.)*.001 - cos(p_ljaw.z*19.)*.006;//distortion
        ljaw = sMax(p_ljaw.y+.93,ljaw,.02);//bottom cut plane
    d = sMin(ljaw, d,.002);//union mandible and d 

    // EYE HOLES
    vec3 p_eyeH = p_skull;
        p_eyeH += sin(p_eyeH.x*29.+cos(p_eyeH.y*32.))*.005; //eye distortion
        float eyes = Ellipsoid(vec3(abs(p_eyeH.x),p_eyeH.y-.4,p_eyeH.z)+vec3(-.29,.49,-1.1),vec3(.21,.25,.25)); // eye balls
        float eyeH = sMin(eyes,Sphere(vec3(abs(p_skull.x),p_skull.yz)-vec3(.25,0.,.7),.35),.05);// eye holes back
        eyeH = sMax(-p_eyeH.y,eyeH,.2);
    d = sMax(eyeH, d,.05); //eye ball subtraction

    // PLANE
    vec3 pPla = p;
        pPla.z += sin(p.y*.2-t*.7)-.2;
        float pla = pPla.z;    
    d = sMin(d,pla,.8);
    
    // EYEBALLS
    vec3 p_eye = p_skull;
    p_eye.x = abs(p_eye.x);
    p_eye.y -= .4;
    p_eye += vec3(-.29,.57,-.9);
    eyes = Ellipsoid(p_eye,vec3(.2));
    
    // UPPER TEETH
    vec3 p_tooth = p_skull;
        p_tooth -= vec3(0,-.77,.7);
        p_tooth *= vec3(1.2,1,1);
        pModPolar(p_tooth.xz, 32.0);

        float teeth = Ellipsoid(p_tooth - vec3(0.43, 0., 0.), vec3(0.03, 0.15, 0.045));
        teeth = max(teeth, -p_skull.y-.73+sin(p_skull.x*32.)*.006);
        teeth = max(teeth, -p_skull.z+.7);
        teeth = sMax(Sphere(p_skull-vec3(0.02,-.88,.98),.23),teeth,.01);// narrowing front teeth
    d = min(d,teeth);
        
    // LOWER TEETH
    vec3 p_ltooth = pN;
        p_ltooth -= vec3(0,-.77,.7);
        p_ltooth *= vec3(1.2,1,1);
        pModPolar(p_ltooth.xz, 32.0);

        float lteeth = Ellipsoid(p_ltooth - vec3(0.42, 0., 0.), vec3(0.03, 0.15, 0.045));
        lteeth = max(lteeth, pN.y+.79+sin(p_skull.x*29.)*.004);
        lteeth = max(lteeth, -pN.z+.7);
        lteeth = sMax(Sphere(pN-vec3(0.005,-.87,.89),.24),lteeth,.02);// narrowing front teeth
    d = min(d,lteeth);
    
    // MATERIALS
    vec2 res = vec2(d,0);
    if(eyes<d) res = vec2(eyes,1);
    
    return res;
}

vec2 RM(vec3 ro, vec3 rd){
    vec2 res = vec2(0);
    float mat = 0.;
    float t=0.;
    for(int i=0;i<MAX_STEPS; i++){
        vec3 p = ro + rd*t;
        float d = map(p).x;
        mat = map(p).y;
        t += d;
        if(t>MAX_DIST||abs(d)<SURF_DIST) break;
    }
    res = vec2(t,mat);
    
    return res;
}

vec3 calcNormal (vec3 p){
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=min(iFrame,0); i<4; i++ ){
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+.001*e).x;
    }
    return normalize(n);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
    float t = t(iTime);
	vec2 m = (iMouse.xy-.5*iResolution.xy)/iResolution.y;
    vec3 ro = vec3(0, 0, 4.);
    ro.xz *= Rot(-m.x*.5);
    ro.yz *= Rot(m.y);
    vec3 rd = GetRayDir(uv, ro, vec3(0,0.,0), 1.);
    
    vec3 col = vec3(1);
    float d = RM(ro, rd).x;
    float mat = RM(ro,rd).y;
    
    if(d<MAX_DIST){
        
        vec3 p = ro + rd*d;
        vec3 n = calcNormal(p);
        float fresnel = pow(1.+dot(rd, n),2.);
        col = vec3(0.2,.3,.4);
        col += fresnel;
        
        // COLOR EYES
        if(mat==1.)col += .7;
    }
    
    col = mix( col, vec3(1), 1.-exp(-0.0002*pow(d,3.)) );//fog    
    col = pow(col, vec3(.4545)); //gamma correction
    
    fragColor = vec4(col,1.);
}


