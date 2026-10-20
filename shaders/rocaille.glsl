// https://www.shadertoy.com/view/WXyczK
/*
    "Rocaille" by @XorDev
    
    This time I added multiple layers of turbulence
    with time and color offsets. Loved the shapes.
    
    -1 Thanks to GregRostami
*/
void mainImage(out vec4 O, vec2 I)
{
    //Vector for scaling and turbulence
    vec2 v = iResolution.xy,
    //Centered and scaled coordinates
    p = (I+I-v)/v.y/.3;
    
    //Iterators for layers and turbulence frequency
    float i, f;
    for(O*=i;i++<9.;
        //Add coloring, attenuating with turbulent coordinates
        O += (cos(i+vec4(0,1,2,3))+1.)/6./length(v))
        //Turbulence loop
        //https://mini.gmshaders.com/p/turbulence
        for(v=p,f=0.;f++<9.;v+=sin(v.yx*f+i+iTime)/f);
    
    //Tanh tonemapping
    //https://www.shadertoy.com/view/ms3BD7
    O = tanh(O*O);
}
