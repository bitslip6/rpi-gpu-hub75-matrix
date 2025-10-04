#define hash(x) fract(sin(x) * 43758.5453123)
vec3 pal(float t){return .5+.5*cos(6.28*(1.*t+vec3(.0,.1,.1)));}
 float stepNoise(float x, float n) { // From Kamoshika shader
   const float factor = 0.3;
   float i = floor(x);
   float f = x - i;
   float u = smoothstep(0.5 - factor, 0.5 + factor, f);
   float res = mix(floor(hash(i) * n), floor(hash(i + 1.) * n), u);
   res /= (n - 1.) * 0.5;
   return res - 1.;
 }
 vec3 path(vec3 p){
   
      vec3 o = vec3(0.);
       o.x += stepNoise(p.z*.05,5.)*5.;
      o.y += stepNoise(p.z*.07,3.975)*5.;
     return o;
   }
   float diam2(vec2 p,float s){p=abs(p); return (p.x+p.y-s)*inversesqrt(3.);}
   vec3 erot(vec3 p,vec3 ax,float t){return mix(dot(ax,p)*ax,p,cos(t))+cross(ax,p)*sin(t);}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;

 vec3 col = vec3(0.);
  
  vec3 ro = vec3(0.,0.,-1.),rt=vec3(0.);
  ro.z+=iTime*5.;
  rt.z += iTime*5.;
  ro+=path(ro);
    rt+=path(rt);
  vec3 z = normalize(rt-ro);
  vec3 x = vec3(z.z,0.,-z.x);
  float i=0.,e=0.,g=0.;
  vec3 rd = mat3(x,cross(z,x),z)*erot(normalize(vec3(uv,1.)),vec3(0.,0.,1.),stepNoise(iTime+hash(uv.x*uv.y*iTime)*.05,6.));
  for(;i++<99.;){
     vec3 p= ro+rd*g;

    p-=path(p);
    float r = 0.;;
    vec3 pp=p;
    float sc=1.;
    for(float j=0.;j++<4.;){
        r = clamp(r+abs(dot(sin(pp*3.),cos(pp.yzx*2.))*.3-.1)/sc,-.5,.5);
        pp=erot(pp,normalize(vec3(.1,.2,.3)),.785+j);
        pp+=pp.yzx+j*50.;
        sc*=1.5;
        pp*=1.5;
      }
      
     float h = abs(diam2(p.xy,7.))-3.-r;
   
     p=erot(p,vec3(0.,0.,1.),path(p).x*.5+p.z*.2);
    float t = length(abs(p.xy)-.5)-.1;
     h= min(t,h);
     g+=e=max(.001,t==h ?abs(h):(h));
     col +=(t==h ?vec3(.3,.2,.1)*(100.*exp(-20.*fract(p.z*.25+iTime)))*mod(floor(p.z*4.)+mod(floor(p.y*4.),2.),2.) :vec3(.1))*.0325/exp(i*i*e);;
    }
    col = mix(col,vec3(.9,.9,1.1),1.-exp(-.01*g*g*g));
    // Output to screen
    fragColor = vec4(col,1.0);
}
