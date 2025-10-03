// Protean clouds by nimitz (twitter: @stormoid)
// CC BY-NC-SA 3.0

#define MARCH_STEPS 110
#define NOISE_OCTAVES 5

// New: toggle this optimization (2 = use previous-step density for lighting)
#define OPT_LIGHT_FROM_PREV 1

#define OPT_COARSE_PREPASS 1   // set to 0 to disable quickly

#define PREPASS_STEPS 5        // small, cheap loop

#define PREPASS_DT_MIN 0.25    // bigger steps than main march

#define PREPASS_DT_MAX 0.45

#define PREPASS_THRESH 0.55    // stop prepass when close to density

#define ALPHA_BREAK 0.97


#ifndef OPT_ADAPTIVE_OCTAVES
#define OPT_ADAPTIVE_OCTAVES 1   // set to 0 to disable quickly
#endif

#ifndef OCT_GATE
#define OCT_GATE 3               // compute first 3 octaves always, gate 4–5
#endif

#ifndef ADAPT_THRESH
#define ADAPT_THRESH 0.52        // conservative: below this, skip remaining octaves
#endif



#ifndef OPT_STEP_TWEAK
#define OPT_STEP_TWEAK 1      // set to 0 to restore original stepping
#endif

#ifndef STEP_MIN
#define STEP_MIN 0.10         // original 0.09
#endif
#ifndef STEP_MAX
#define STEP_MAX 0.32         // original 0.30
#endif
#ifndef STEP_A
#define STEP_A 0.52           // original 0.50
#endif
#ifndef STEP_B
#define STEP_B 0.045          // original 0.05
#endif




mat2 rot(in float a){float c = cos(a), s = sin(a);return mat2(c,s,-s,c);}
const mat3 m3 = mat3(0.33338, 0.56034, -0.71817, -0.87887, 0.32651, -0.15323, 0.15162, 0.69596, 0.61339)*1.93;
float mag2(vec2 p){return dot(p,p);}
float linstep(in float mn, in float mx, in float x){ return clamp((x - mn)/(mx - mn), 0., 1.); }
float prm1 = 0.;
vec2 bsMo = vec2(0);

vec2 disp(float t){ return vec2(sin(t*0.22)*1., cos(t*0.175)*1.)*2.; }



vec2 map(vec3 p)
{
    vec3 p2 = p;
    p2.xy -= disp(p.z).xy;
    p.xy *= rot(sin(p.z+iTime)*(0.1 + prm1*0.05) + iTime*0.09);

    float cl = dot(p2.xy, p2.xy);   // same as mag2(p2.xy)
    float d = 0.0;
    p *= 0.61;
    float z = 1.0;
    float trk = 1.0;
    float dspAmp = 0.1 + prm1*0.2;

    for (int i = 0; i < NOISE_OCTAVES; i++)
    {
        p += sin(p.zxy*0.75*trk + iTime*trk*0.8) * dspAmp;
        d -= abs(dot(cos(p), sin(p.yzx)) * z);
        z   *= 0.57;
        trk *= 1.4;
        p = p * m3;

#if OPT_ADAPTIVE_OCTAVES
        // After OCT_GATE octaves, estimate the final density and bail if it's “empty”
        if (i + 1 == OCT_GATE) {
            float d_est = abs(d + prm1*3.0) + prm1*0.3 - 2.5 + bsMo.y;
            d_est += cl*0.2 + 0.25;             // same post terms used below
            if (d_est < ADAPT_THRESH) {
                // Skip remaining octaves, they won't change step/shade decisions much
                break;
            }
        }
#endif
    }

    d = abs(d + prm1*3.0) + prm1*0.3 - 2.5 + bsMo.y;
    return vec2(d + cl*0.2 + 0.25, cl);
}





vec4 render( in vec3 ro, in vec3 rd, float time )
{
    vec4 rez = vec4(0);
    const float ldst = 8.;
    vec3 lpos = vec3(disp(time + ldst)*0.5, time + ldst);
    float t = 1.5;
    float fogT = 0.;


#if OPT_COARSE_PREPASS
    // coarse skip through empty space
    for (int k = 0; k < PREPASS_STEPS; k++) {
        vec3 p = ro + t * rd;
        float d = map(p).x;
        if (d > PREPASS_THRESH) break;

        float dn = clamp(d + 2.0, 0.0, 3.0);
        t += clamp(0.6 - dn*dn*0.05, PREPASS_DT_MIN, PREPASS_DT_MAX);
    }
#endif

#if OPT_LIGHT_FROM_PREV
    // seed previous-step density for cheap lighting
    float prev_mpvx = map(ro + t*rd - rd*0.6).x;
#endif



    for(int i=0; i<MARCH_STEPS; i++)
    {
        if(rez.a > ALPHA_BREAK) break;

        vec3 pos = ro + t*rd;
        vec2 mpv = map(pos);
        float den = clamp(mpv.x-0.3,0.,1.)*1.12;
        float dn  = clamp((mpv.x + 2.),0.,3.);
        
        vec4 col = vec4(0);
        if (mpv.x > 0.6)
        {
            col = vec4(sin(vec3(5.,0.4,0.2) + mpv.y*0.1 + sin(pos.z*0.4)*0.5 + 1.8)*0.5 + 0.5,0.08);
            col *= den*den*den;
            col.rgb *= linstep(4.,-2.5, mpv.x)*2.3;

#if OPT_LIGHT_FROM_PREV
            // New: reuse previous-step density as a cheap gradient proxy
            float grad = prev_mpvx - mpv.x;                  // >0 when entering denser region
            float dif  = clamp(grad*0.8 + 0.35, 0.001, 1.0); // tuned scale/bias
#else
            // Baseline from your last step: single forward sample
            float dif = clamp((den - map(pos + rd*0.6).x)/5.0, 0.001, 1.0);
#endif
            col.xyz *= den*(vec3(0.005,.045,.075) + 1.5*vec3(0.033,0.07,0.03)*dif);
        }
        
        //float fogC = exp(t*0.2 - 2.2);
		const float inv_ln2 = 1.4426950408889634;   // 1 / ln(2)
		float fogC = exp2((t*0.2 - 2.2) * inv_ln2);

        col.rgba += vec4(0.06,0.11,0.11, 0.1)*clamp(fogC-fogT, 0., 1.);
        fogT = fogC;
        rez = rez + col*(1. - rez.a);
        // t += clamp(0.5 - dn*dn*.05, 0.10, 0.31);
		#if OPT_STEP_TWEAK
			t += clamp(STEP_A - dn*dn*STEP_B, STEP_MIN, STEP_MAX);
		#else
			t += clamp(0.5 - dn*dn*.05, 0.09, 0.3);
		#endif


#if OPT_LIGHT_FROM_PREV
        prev_mpvx = mpv.x; // advance light proxy
#endif
    }
    return clamp(rez, 0.0, 1.0);
}

float getsat(vec3 c)
{
    float mi = min(min(c.x, c.y), c.z);
    float ma = max(max(c.x, c.y), c.z);
    return (ma - mi)/(ma+ 1e-7);
}

vec3 iLerp(in vec3 a, in vec3 b, in float x)
{
    vec3 ic = mix(a, b, x) + vec3(1e-6,0.,0.);
    float sd = abs(getsat(ic) - mix(getsat(a), getsat(b), x));
    vec3 dir = normalize(vec3(2.*ic.x - ic.y - ic.z, 2.*ic.y - ic.x - ic.z, 2.*ic.z - ic.y - ic.x));
    float lgt = dot(vec3(1.0), ic);
    float ff = dot(dir, normalize(ic));
    ic += 1.5*dir*sd*ff*lgt;
    return clamp(ic,0.,1.);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{   
    vec2 q = fragCoord.xy/iResolution.xy;
    vec2 p = (gl_FragCoord.xy - 0.5*iResolution.xy)/iResolution.y;
    bsMo = (iMouse.xy - 0.5*iResolution.xy)/iResolution.y;
    
    float time = iTime*3.;
    vec3 ro = vec3(0,0,time);
    ro += vec3(sin(iTime)*0.5,sin(iTime*1.)*0.,0);
        
    float dspAmp = .85;
    ro.xy += disp(ro.z)*dspAmp;
    float tgtDst = 3.5;
    
    vec3 target = normalize(ro - vec3(disp(time + tgtDst)*dspAmp, time + tgtDst));
    ro.x -= bsMo.x*2.;
    vec3 rightdir = normalize(cross(target, vec3(0,1,0)));
    vec3 updir = normalize(cross(rightdir, target));
    rightdir = normalize(cross(updir, target));
    vec3 rd=normalize((p.x*rightdir + p.y*updir)*1. - target);
    rd.xy *= rot(-disp(time + 3.5).x*0.2 + bsMo.x);
    prm1 = smoothstep(-0.4, 0.4,sin(iTime*0.3));
    vec4 scn = render(ro, rd, time);
        
    vec3 col = scn.rgb;
    col = iLerp(col.bgr, col.rgb, clamp(1.-prm1,0.05,1.));
    col = pow(col, vec3(.55,0.65,0.6))*vec3(1.,.97,.9);
    col *= pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.12)*0.7+0.3; //Vign
    fragColor = vec4( col, 1.0 );
}

