import numpy as np
import plotly.graph_objs as go


def moduloAB(x, a, b): #maps a real number onto the unit circle identified with 
                       #the interval [a,b), b-a=2*PI
        if a>=b:
            raise ValueError('Incorrect interval ends')
        y=(x-a)%(b-a)
        return y+b if y<0 else y+a

def test_2PI(x):
    return 0<= x <2*PI

def get_ideogram_ends(ideogram_len, gap):
    ideo_ends=[]
    left=0
    for k in range(len(ideogram_len)):
        right=left+ideogram_len[k]
        ideo_ends.append([left, right])
        left=right+gap
    return ideo_ends

def check_data(data_matrix):
    L, M=data_matrix.shape
    if L!=M:
        raise ValueError('Data array must have (n,n) shape')
    return L

def make_ideogram_arc(R, phi, a=50):
    # R is the circle radius
    # phi is the list of ends angle coordinates of an arc
    # a is a parameter that controls the number of points to be evaluated on an arc
    if not test_2PI(phi[0]) or not test_2PI(phi[1]):
        phi=[moduloAB(t, 0, 2*PI) for t in phi]
    length=(phi[1]-phi[0])% 2*PI
    nr=5 if length<=PI/4 else int(a*length/PI)

    if phi[0] < phi[1]:
        theta=np.linspace(phi[0], phi[1], nr)
    else:
        phi=[moduloAB(t, -PI, PI) for t in phi]
        theta=np.linspace(phi[0], phi[1], nr)
    return R*np.exp(1j*theta)

def make_layout(title, plot_size):
    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

    return go.Layout(title=title,
                  xaxis=dict(axis),
                  yaxis=dict(axis),
                  showlegend=False,
                  width=plot_size,
                  height=plot_size,
                  margin=dict(t=25, b=25, l=25, r=25),
                  hovermode='closest',
                  shapes=list(), # to this list one appends below the dicts defining the ribbon,
                           #respectively the ideogram shapes
                 )

def make_ribbon_arc(theta0, theta1):

    if test_2PI(theta0) and test_2PI(theta1):
        if theta0 < theta1:
            theta0= moduloAB(theta0, -PI, PI)
            theta1= moduloAB(theta1, -PI, PI)
            if theta0*theta1>0:
                raise ValueError('incorrect angle coordinates for ribbon')

        nr=int(40*(theta0-theta1)/PI)
        if nr<=2: nr=3
        theta=np.linspace(theta0, theta1, nr)
        pts=np.exp(1j*theta)# points on arc in polar complex form

        string_arc=''
        for k in range(len(theta)):
            string_arc+='L '+str(pts.real[k])+', '+str(pts.imag[k])+' '
        return   string_arc
    else:
        raise ValueError('the angle coordinates for an arc side of a ribbon must be in [0, 2*pi]')

def make_ribbon(l, r, line_color, fill_color, radius=0.2):
    #l=[l[0], l[1]], r=[r[0], r[1]]  represent the opposite arcs in the ribbon 
    #line_color is the color of the shape boundary
    #fill_color is the fill color for the ribbon shape
    poligon=ctrl_rib_chords(l,r, radius)
    b,c =poligon
    # print(list(c))
    # print(list(c)[::-1])
    return  dict(
                line=dict(
                color=line_color, width=0.5
            ),
            path=  make_q_bezier(b)+make_ribbon_arc(r[0], r[1])+
                   make_q_bezier(list(c)[::-1])+make_ribbon_arc(l[1], l[0]), 
                #    make_q_bezier(c)+make_ribbon_arc(l[1], l[0]),
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

def make_self_rel(l, line_color, fill_color, radius):
    #radius is the radius of Bezier control point b_1
    b=control_pts([l[0], (l[0]+l[1])/2, l[1]], radius)
    return  dict(
                line=dict(
                color=line_color, width=0.5
            ),
            path=  make_q_bezier(b)+make_ribbon_arc(l[1], l[0]),
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

def invPerm(perm):
    # function that returns the inverse of a permutation, perm
    inv = [0] * len(perm)
    for i, s in enumerate(perm):
        inv[s] = i
    return inv

def make_ideo_shape(path, line_color, fill_color):
    #line_color is the color of the shape boundary
    #fill_collor is the color assigned to an ideogram
    return  dict(
                  line=dict(
                  color=line_color,
                  width=0.45
                 ),

            path=  path,
            type='path',
            fillcolor=fill_color,
            layer='below'
        )

def map_data(data_matrix, row_value, ideogram_length):
    mapped=np.zeros(data_matrix.shape)
    for j  in range(L):
        mapped[:, j]=ideogram_length*data_matrix[:,j]/row_value
    return mapped

def make_ribbon_ends(mapped_data, ideo_ends,  idx_sort):
    L=mapped_data.shape[0]
    ribbon_boundary=np.zeros((L,L+1))
    for k in range(L):
        start=ideo_ends[k][0]
        ribbon_boundary[k][0]=start
        for j in range(1,L+1):
            J=idx_sort[k][j-1]
            ribbon_boundary[k][j]=start+mapped_data[k][J]
            start=ribbon_boundary[k][j]
    return [[(ribbon_boundary[k][j],ribbon_boundary[k][j+1] ) for j in range(L)] for k in range(L)]

def control_pts(angle, radius):     # compute the control points
    #angle is a  3-list containing angular coordinates of the control points b0, b1, b2
    #radius is the distance from b1 to the  origin O(0,0) 

    if len(angle)!=3:
        raise InvalidInputError('angle must have len =3')
    b_cplx=np.array([np.exp(1j*angle[k]) for k in range(3)])
    b_cplx[1]=radius*b_cplx[1]
    return zip(b_cplx.real, b_cplx.imag)

def make_q_bezier(b):# defines the Plotly SVG path for a quadratic Bezier curve defined by the 
                     #list of its control points
    # if len(list(b))!=3:
    #     raise valueError('control poligon must have 3 points')
    A, B, C=b
    # print(A,B,C)

    return 'M '+str(A[0])+',' +str(A[1])+' '+'Q '+\
                str(B[0])+', '+str(B[1])+ ' '+\
                str(C[0])+', '+str(C[1])

def ctrl_rib_chords(l, r, radius):
    # this function returns a 2-list containing control poligons of the two quadratic Bezier
    #curves that are opposite sides in a ribbon
    #l (r) the list of angular variables of the ribbon arc ends defining 
    #the ribbon starting (ending) arc 
    # radius is a common parameter for both control polygons
    if len(l)!=2 or len(r)!=2:
        raise ValueError('the arc ends must be elements in a list of len 2')
    return [control_pts([l[j], (l[j]+r[j])/2, r[j]], radius) for j in range(2)]

def basic_set_1():
    matrix=np.array([[16,  3, 28,  0, 18],
                    [18,  0, 12,  5, 29],
                    [ 9, 11, 17, 27,  0],
                    [19,  0, 31, 11, 12],
                    [23, 17, 10,  0, 34]], dtype=int)
    labels=['Emma', 'Isabella', 'Ava', 'Olivia', 'Sophia']
    ideo_colors=['rgba(72, 61, 139, 0.75)',
                'rgba(106, 90, 205, 0.75)',
                'rgba(123, 104, 238, 0.75)',
                'rgba(132, 112, 255, 0.75)',
                'rgba(93, 71, 139, 0.75)']
    radii_sribb=[0.4, 0.30, 0.35, 0.39, 0.12, 0.1, 0.1, 0.1]

    return matrix,labels,ideo_colors,radii_sribb

def basic_set_2():
    matrix = np.arange(64).reshape(8, 8)
    labels=['Emma', 'Isabella', 'Ava', 'Olivia', 'Sophia','aaa','bbb','ccc']
    ideo_colors=['rgba(72, 61, 139, 0.75)',
                'rgba(106, 90, 205, 0.75)',
                'rgba(123, 104, 238, 0.75)',
                'rgba(132, 112, 255, 0.75)',
                'rgba(143, 188, 143, 0.75)',
                'rgba(46, 139, 87, 0.75)',
                'rgba(60, 179, 113, 0.75)',
                'rgba(93, 71, 139, 0.75)']
    radii_sribb=[0.4, 0.30, 0.35, 0.39, 0.12, 0.1, 0.1, 0.1]

    return matrix,labels,ideo_colors,radii_sribb

if __name__ == "__main__":
    ### basic setting
    matrix,labels,ideo_colors,radii_sribb = basic_set_2()
    layout=make_layout('Chord diagram', 400)

    ### draw circle
    PI=np.pi
    gap=2*PI*0.005
    L=check_data(matrix)
    row_sum=[np.sum(matrix[k,:]) for k in range(L)]
    ideogram_length=2*PI*np.asarray(row_sum)/sum(row_sum)-gap*np.ones(L)
    ideo_ends=get_ideogram_ends(ideogram_length, gap)

    ideograms=[]
    for k in range(len(ideo_ends)):
        z= make_ideogram_arc(1.1, ideo_ends[k])
        zi=make_ideogram_arc(1.0, ideo_ends[k])
        m=len(z)
        n=len(zi)
        ideograms.append(go.Scatter(x=z.real,
                                y=z.imag,
                                mode='lines',
                                line=dict(color=ideo_colors[k], shape='spline', width=0.5),
                                text=labels[k]+'<br>'+'{:d}'.format(row_sum[k]),
                                hoverinfo='text'
                                )
                        )


        path='M '
        for s in range(m):
            path+=str(z.real[s])+', '+str(z.imag[s])+' L '

        Zi=np.array(zi.tolist()[::-1])

        for s in range(m):
            path+=str(Zi.real[s])+', '+str(Zi.imag[s])+' L '
        path+=str(z.real[0])+' ,'+str(z.imag[0])
        # print(path)
        # layout['shapes'].append(make_ideo_shape(path,'rgb(150,150,150)' , ideo_colors[k]))
        layout['shapes'] = layout['shapes']+(make_ideo_shape(path,'rgb(150,150,150)' , ideo_colors[k]),)

    ### draw ribbon
    mapped_data=map_data(matrix, row_sum, ideogram_length)
    idx_sort=np.argsort(mapped_data, axis=1)
    ribbon_ends=make_ribbon_ends(mapped_data, ideo_ends,  idx_sort)
    ribbon_color=[L*[ideo_colors[k]] for k in range(L)]
    ribbon_color[0][4]=ideo_colors[4]
    ribbon_color[1][2]=ideo_colors[2]
    ribbon_color[2][3]=ideo_colors[3]
    ribbon_color[2][4]=ideo_colors[4]

    ribbon_info=[]

    for k in range(L):

        sigma=idx_sort[k]
        sigma_inv=invPerm(sigma)
        for j in range(k, L):
            if matrix[k][j]==0 and matrix[j][k]==0: continue
            eta=idx_sort[j]
            eta_inv=invPerm(eta)
            l=ribbon_ends[k][sigma_inv[j]]

            if j==k:
                print(k)
                layout['shapes'] = layout['shapes']+(make_self_rel(l, 'rgb(175,175,175)' ,
                                        ideo_colors[k], radius=radii_sribb[k]),)
                z=0.9*np.exp(1j*(l[0]+l[1])/2)

                #the text below will be displayed when hovering the mouse over the ribbon
                text=labels[k]+' commented on '+ '{:d}'.format(matrix[k][k])+' of '+ 'herself Fb posts',
                ribbon_info.append(go.Scatter(x=[z.real],
                                        y=[z.imag],
                                        mode='markers',
                                        marker=dict(size=0.5, color=ideo_colors[k]),
                                        text=text,
                                        hoverinfo='text'
                                        )
                                )
            else:
                r=ribbon_ends[j][eta_inv[k]]
                zi=0.9*np.exp(1j*(l[0]+l[1])/2)
                zf=0.9*np.exp(1j*(r[0]+r[1])/2)
                #texti and textf are the strings that will be displayed when hovering the mouse 
                #over the two ribbon ends
                texti=labels[k]+' commented on '+ '{:d}'.format(matrix[k][j])+' of '+\
                    labels[j]+ ' Fb posts',

                textf=labels[j]+' commented on '+ '{:d}'.format(matrix[j][k])+' of '+\
                labels[k]+ ' Fb posts',
                ribbon_info.append(go.Scatter(x=[zi.real],
                                        y=[zi.imag],
                                        mode='markers',
                                        marker=dict(size=0.5, color=ribbon_color[k][j]),
                                        text=texti,
                                        hoverinfo='text'
                                        )
                                ),
                # print(ribbon_info)
                ribbon_info.append(go.Scatter(x=[zf.real],
                                        y=[zf.imag],
                                        mode='markers',
                                        marker=dict(size=0.5, color=ribbon_color[k][j]),
                                        text=textf,
                                        hoverinfo='text'
                                        )
                                )
                r=(r[1], r[0])#IMPORTANT!!!  Reverse these arc ends because otherwise you get
                            # a twisted ribbon
                #append the ribbon shape
                layout['shapes'] = layout['shapes']+(make_ribbon(l, r, 'rgb(175,175,175)' , ribbon_color[k][j]),)

    ### constitute the diagram
    data = go.Data(ideograms+ribbon_info)
    fig = go.Figure(data=data, layout=layout)
    fig.show()

