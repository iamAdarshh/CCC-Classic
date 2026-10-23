// Six-emitter beam search with exact one-tick lookahead and parent diversity.
// All candidates are independently replayed before publication by campaign.py.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <random>
#include <cmath>
using namespace std;
static uint64_t hash64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
template<int W>struct Bit{array<uint64_t,W>a{};void set(int i){a[i/64]|=1ULL<<(i%64);}};
template<int W>struct Group{Bit<W>one,two,three,latent,latenttwo;};
template<int W>struct State{
 Bit<W>matter;array<uint8_t,6>state{};array<char,6>op{};
 int count=0,path=-1;uint64_t hash=0;double score=0,base=0;
};
struct Path{int parent;array<char,6>op;};
const int Q[6]={1,0,-1,-1,0,1},S[6]={0,1,1,0,-1,-1};const char OP[4]={'-','F','B','T'};

template<int W>class Planner{
 int radius,P,required,width,maxsteps,variant,parentcap,lookfactor,targetsize;
 uint64_t seed;double seconds;Bit<W>target,forbidden;vector<pair<int,int>>cells,rail;
 vector<Bit<W>>rays[6],latent[6];vector<Path>paths;
 vector<Bit<W>>reach1[6];Bit<W>weighted;
 vector<vector<uint8_t>>distances[6];
 vector<array<uint8_t,6>>initials;
 int next[144][4];chrono::steady_clock::time_point start;
 array<uint8_t,6>cycle_start{};
 bool cyclemode=false;
 vector<Bit<W>>reach2[6];
 vector<vector<Path>>unused;
 bool expired(){return chrono::duration<double>(chrono::steady_clock::now()-start).count()>seconds;}
 double projected_distance(const State<W>&node) {
  // Small per-pair BFS can be used as a guaranteed fallback; for the beam
  // heuristic use the lower bound from physical emitter travel instead.
  return distance_score(node);
 }
 void groups(const State<W>&node,Group<W>(&g)[2][64]){
  for(int side=0;side<2;side++)for(int code=0;code<64;code++){
   auto &out=g[side][code];
   for(int j=0;j<3;j++){
    int e=side*3+j,st=next[node.state[e]][(code>>(2*j))&3];
    for(int w=0;w<W;w++){
     uint64_t r=rays[e][st].a[w],l=latent[e][st].a[w];
     out.three.a[w]|=out.two.a[w]&r;out.two.a[w]|=out.one.a[w]&r;out.one.a[w]|=r;
     out.latenttwo.a[w]|=out.latent.a[w]&l;out.latent.a[w]|=l;
    }
   }
  }
 }
 double future(const State<W>&node){
  Group<W>g[2][64];groups(node,g);
  double best=0;
  for(int a=0;a<64;a++)for(int b=0;b<64;b++){
   auto&x=g[0][a];auto&y=g[1][b];int gain=0,latentcount=0;bool invalid=false;
   for(int w=0;w<W;w++){
    uint64_t two=x.two.a[w]|y.two.a[w]|(x.one.a[w]&y.one.a[w]);
    uint64_t three=x.three.a[w]|y.three.a[w]|(x.two.a[w]&y.one.a[w])|(x.one.a[w]&y.two.a[w]);
    uint64_t made=two&~three;
    if((made&forbidden.a[w])||(three&node.matter.a[w])){invalid=true;break;}
    gain+=__builtin_popcountll(made&target.a[w]&~node.matter.a[w]);
    if(variant>=8) {
     uint64_t latent=x.latenttwo.a[w]|y.latenttwo.a[w]|(x.latent.a[w]&y.latent.a[w]);
     latentcount+=__builtin_popcountll(latent&target.a[w]&~(node.matter.a[w]|made));
    }
   }
   if(!invalid)best=max(best,gain+latentcount*.03);
  }
  return best;
 }
 double future2(const State<W>&node) {
  Group<W>g[2][64];groups(node,g);
  struct Choice{State<W>state;double score;int gain;};
  vector<Choice>choices;
  int originalmade=0;for(int w=0;w<W;w++)originalmade+=__builtin_popcountll(node.matter.a[w]);
  for(int a=0;a<64;a++)for(int b=0;b<64;b++){
   auto&x=g[0][a];auto&y=g[1][b];State<W>child=node;int gain=0,lat=0;bool invalid=false;
   for(int w=0;w<W;w++){
    uint64_t two=x.two.a[w]|y.two.a[w]|(x.one.a[w]&y.one.a[w]);
    uint64_t three=x.three.a[w]|y.three.a[w]|(x.two.a[w]&y.one.a[w])|(x.one.a[w]&y.two.a[w]);
    uint64_t made=two&~three;
    if((made&forbidden.a[w])||(three&node.matter.a[w])){invalid=true;break;}
    child.matter.a[w]=node.matter.a[w]|(made&target.a[w]);
    gain+=__builtin_popcountll(made&target.a[w]&~node.matter.a[w]);
    uint64_t latent=x.latenttwo.a[w]|y.latenttwo.a[w]|(x.latent.a[w]&y.latent.a[w]);
    lat+=__builtin_popcountll(latent&target.a[w]&~child.matter.a[w]);
   }
   if(invalid)continue;
   if(originalmade+gain==targetsize) {child.matter={};child.count++;}
   for(int e=0;e<6;e++){int op=((e<3?a:b)>>(2*(e%3)))&3;child.state[e]=next[node.state[e]][op];}
   choices.push_back({child,gain+lat*.03,gain});
  }
  int take=min(8,(int)choices.size());
  partial_sort(choices.begin(),choices.begin()+take,choices.end(),[](auto&a,auto&b){return a.score>b.score;});
  double best=0;
  for(int i=0;i<take;i++)best=max(best,choices[i].gain+future(choices[i].state));
  return best;
 }
 double configurationscore(const State<W>&node) {
  // Shape-aware travel pressure: prefer making the future beam-pair grid cover
  // all remaining cells, not just maximizing a count of nearby cells.
  int worst=0,sum=0;
  for(int i=0;i<(int)cells.size();i++) {
   if(!(target.a[i/64]>>(i%64)&1) || (node.matter.a[i/64]>>(i%64)&1))continue;
   array<int,6>d;
   for(int e=0;e<6;e++) {
    d[e]=1000;
    for(int p=0;p<P;p++)if(rays[e][2*p+1].a[i/64]>>(i%64)&1) {
     int distance=abs(node.state[e]/2-p);distance=min(distance,P-distance)+(node.state[e]%2?0:1);d[e]=min(d[e],distance);
    }
   }
   int best=1000;
   for(int e=0;e<6;e++)for(int f=e+1;f<6;f++)if(e%3!=f%3)best=min(best,max(d[e],d[f]));
   worst=max(worst,best);sum+=best;
  }
  return -worst*.4-sum*.03;
 }
 double potential(const State<W>&node) {
  Bit<W>one,two;int value=0;
  for(int e=0;e<6;e++)for(int w=0;w<W;w++){
   uint64_t r=reach2[e][node.state[e]].a[w];two.a[w]|=one.a[w]&r;one.a[w]|=r;
  }
  for(int w=0;w<W;w++)value+=__builtin_popcountll(two.a[w]&target.a[w]&~node.matter.a[w]);
  return value*.1;
 }
 double distance_score(const State<W>&node) {
  // Exact per-cell earliest arrival ignoring interactions with other cells.
  // The maximum emphasizes hard-to-reach missing cells before they become a tail.
  int total=0,worst=0;
  for(int i=0;i<(int)cells.size();i++) {
   if(!(target.a[i/64]>>(i%64)&1) || (node.matter.a[i/64]>>(i%64)&1))continue;
   int first=1000,second=1000;
   for(int e=0;e<6;e++) {
    int closest=distances[e][node.state[e]][i];
    if(closest<first){second=first;first=closest;}else if(closest<second)second=closest;
   }
   total+=second;worst=max(worst,second);
  }
  const double weight[4]={.05,.1,.2,.4};
  return -total*weight[variant%4]-worst*.3;
 }
 void output(int path,const array<char,6>&op){
  vector<array<char,6>>program{op};for(int p=path;p>=0;p=paths[p].parent)program.push_back(paths[p].op);
  reverse(program.begin(),program.end());for(auto&line:program){for(int e=0;e<6;e++)cout<<(e?" ":"")<<line[e];cout<<'\n';}
 }
public:
 Planner(int r,int count,int beamwidth,int steps,int strategy,int cap,int look,uint64_t sd,double secs,const vector<string>&pattern):radius(r),P(6*r),required(count),width(beamwidth),maxsteps(steps),variant(strategy),parentcap(cap),lookfactor(look),seed(sd),seconds(secs){
  start=chrono::steady_clock::now();
  pair<int,int>corners[6]={{-r,0},{0,-r},{r,-r},{r,0},{0,r},{-r,r}};
  for(int e=0;e<6;e++){auto[q,s]=corners[e];auto[nq,ns]=corners[(e+1)%6];for(int k=0;k<r;k++)rail.emplace_back(q+k*(nq-q)/r,s+k*(ns-s)/r);}
  for(int q=1-r;q<r;q++)for(int s=1-r;s<r;s++)if(max({abs(q),abs(s),abs(q+s)})<r)cells.emplace_back(q,s);
  vector<pair<int,int>>tc;
  for(int i=0;i<(int)cells.size();i++){auto[q,s]=cells[i];if(pattern[s+r][2*q+s+2*r]=='M'){target.set(i);tc.push_back(cells[i]);}}
  targetsize=tc.size();
  if(variant>=12 && variant<16)for(int i=0;i<(int)cells.size();i++) {
   if(!(target.a[i/64]>>(i%64)&1))continue;
   auto[q,s]=cells[i];int counts[3]={};
   for(auto[tq,ts]:tc){counts[0]+=s==ts;counts[1]+=q==tq;counts[2]+=q+s==tq+ts;}
   int high=max({counts[0],counts[1],counts[2]});
   if(high<=r)weighted.set(i);
  }
  for(int i=0;i<(int)cells.size();i++){auto[q,s]=cells[i];for(auto[tq,ts]:tc)if(max({abs(q-tq),abs(s-ts),abs(q+s-tq-ts)})==1)forbidden.set(i);}
  for(int w=0;w<W;w++)forbidden.a[w]&=~target.a[w];
  for(int st=0;st<P*2;st++)for(int op=0;op<4;op++)next[st][op]=2*((st/2+(op==1)-(op==2)+P)%P)+((st%2)^(op==3));
  for(int e=0;e<6;e++){
   rays[e].resize(P*2);latent[e].resize(P*2);reach1[e].resize(P*2);
   for(int p=0;p<P;p++){
    auto[eq,es]=rail[p];
    for(int i=0;i<(int)cells.size();i++){auto[q,s]=cells[i];int dq=q-eq,ds=s-es;if(dq*S[e]==ds*Q[e]&&dq*Q[e]+ds*S[e]>0)rays[e][2*p+1].set(i);}
    for(int a=0;a<2;a++)for(int w=0;w<W;w++)latent[e][2*p+a].a[w]=rays[e][2*p+1].a[w]&target.a[w];
   }
   for(int st=0;st<P*2;st++)for(int op=0;op<4;op++)for(int w=0;w<W;w++)reach1[e][st].a[w]|=rays[e][next[st][op]].a[w]&target.a[w];
   if(variant>=48) {
    reach2[e].resize(P*2);
    for(int st=0;st<P*2;st++)for(int op=0;op<4;op++)for(int w=0;w<W;w++)reach2[e][st].a[w]|=reach1[e][next[st][op]].a[w];
   }
   if(variant>=16 && variant<20) {
    distances[e].resize(P*2,vector<uint8_t>(cells.size(),255));
    for(int st=0;st<P*2;st++)for(int i=0;i<(int)cells.size();i++) {
     int best=255;
     for(int p=0;p<P;p++)if(rays[e][2*p+1].a[i/64]>>(i%64)&1) {
      int d=abs(st/2-p);d=min(d,P-d)+(st%2?0:1);best=min(best,d);
     }
     distances[e][st][i]=best;
    }
   }
  }
 }
 int solve(){
   vector<State<W>>beam(1);for(int e=0;e<6;e++)beam[0].state[e]=2*e*radius;
  cyclemode=variant>=60;
  if((variant>=32 && variant<36)||cyclemode) {
    int state;for(int e=0;e<6;e++){cin>>state;beam[0].state[e]=state;}
    for(int w=0;w<W;w++)cin>>beam[0].matter.a[w];
   }
   cycle_start=beam[0].state;
   unordered_set<uint64_t>visited;
   unordered_map<uint64_t,double>priorities;
  for(int depth=1;depth<=maxsteps&&!expired();depth++){
   vector<State<W>>candidates;candidates.reserve(beam.size()*parentcap);
   for(auto&node:beam){
   Group<W>g[2][64];groups(node,g);vector<State<W>>local;
    for(int a=0;a<64;a++)for(int b=0;b<64;b++){
     auto&x=g[0][a];auto&y=g[1][b];State<W>child;int made=0,lat=0,wmade=0;bool invalid=false;
     for(int w=0;w<W;w++){
      uint64_t two=x.two.a[w]|y.two.a[w]|(x.one.a[w]&y.one.a[w]);
      uint64_t three=x.three.a[w]|y.three.a[w]|(x.two.a[w]&y.one.a[w])|(x.one.a[w]&y.two.a[w]);
      uint64_t born=two&~three;
      if((born&forbidden.a[w])||(three&node.matter.a[w])){invalid=true;break;}
      uint64_t m=node.matter.a[w]|(born&target.a[w]);child.matter.a[w]=m;made+=__builtin_popcountll(m);
      uint64_t l=x.latenttwo.a[w]|y.latenttwo.a[w]|(x.latent.a[w]&y.latent.a[w]);
      lat+=__builtin_popcountll(l&~m&target.a[w]);
      if(variant>=12 && variant<16)wmade+=__builtin_popcountll(m&weighted.a[w]);
     }
     if(invalid)continue;
     child.count=node.count;
     if(made==targetsize){child.count++;child.matter={};made=0;}
     uint64_t config=0;
     for(int e=0;e<6;e++){int op=((e<3?a:b)>>(2*(e%3)))&3;child.state[e]=next[node.state[e]][op];child.op[e]=OP[op];config|=uint64_t(child.state[e])<<(8*e);}
      if(child.count>=required&&!cyclemode){output(node.path,child.op);return 0;}
      if(cyclemode) {
       int needed=0;
       for(int e=0;e<6;e++) {
        int d=abs(child.state[e]/2-cycle_start[e]/2);
        needed=max(needed,min(d,P-d)+(child.state[e]%2!=cycle_start[e]%2));
       }
       if(child.count==required&&needed==0){output(node.path,child.op);return 0;}
       if(child.count>required||needed>maxsteps-depth)continue;
      }
     uint64_t h=hash64(config^hash64(child.count));for(int w=0;w<W;w++)h=hash64(h^child.matter.a[w]);child.hash=h;
     if((variant<4 || variant>=8)&&visited.count(h))continue;
     double jitter=(hash64(h+seed)&65535)/65536.0;
     double latentweight=(variant%4==0)?.03:((variant%4==1)?.1:((variant%4==2)?.3:.03));
     child.base=child.count*targetsize+made;
     child.score=child.base+lat*latentweight+jitter*.005;
     if(variant>=36 && variant<40) {
      // Controlled exploration across near-equal configurations, not merely
      // random tie breaking. Independent seeds sample different production paths.
      const double exploration[4]={.1,.25,.5,1.0};
     child.score+=jitter*exploration[variant%4];
     }
     if(variant>=40 && variant<44) {
      // Trade a little immediate gain for configurations whose next tick builds
      // more. Different discounts produce materially different beam frontiers.
      child.score=child.base+lat*.03+jitter*.01;
     }
     if(variant>=44 && variant<48) {
      // Prefer useful emitter pair line intersections even when both emitters
      // are currently off; add slight stochastic exploration at fixed width.
      child.score=child.base+lat*.06+jitter*.05;
     }
     if(variant>=48 && variant<52)child.score+=potential(child);
     if(cyclemode) {
      int sum=0,worst=0;
      for(int e=0;e<6;e++) {
       int d=abs(child.state[e]/2-cycle_start[e]/2);
       d=min(d,P-d)+(child.state[e]%2!=cycle_start[e]%2);
       sum+=d;worst=max(worst,d);
      }
      child.score-=sum*.03+worst*.08;
     }
     if(variant>=52 && variant<56) {
      // Preserve material-set diversity without imposing an arbitrary cap.
      // A small stable random offset for each built subset changes construction
      // ordering across restarts, rather than only emitter-configuration ties.
      uint64_t key=hash64(child.count);
      for(int w=0;w<W;w++)key=hash64(key^child.matter.a[w]);
      double noise=(hash64(key+seed)&65535)/65536.0;
      const double strength[4]={.25,.5,1.0,2.0};child.score+=noise*strength[variant%4];
     }
     if(variant>=12 && variant<16) {
      int totalweighted=0;for(int w=0;w<W;w++)totalweighted+=__builtin_popcountll(weighted.a[w]);
      if(child.count!=node.count)wmade=0;
      child.score+=(child.count*totalweighted+wmade)*.5;
     }
     child.path=node.path;local.push_back(child);
    }
    auto better=[](auto&a,auto&b){return a.score>b.score;};
    int take=min(parentcap*8,(int)local.size());
    partial_sort(local.begin(),local.begin()+take,local.end(),better);
    unordered_set<uint64_t>seen;int kept=0;
    for(int i=0;i<take&&kept<parentcap;i++)if(seen.insert(local[i].hash).second){candidates.push_back(local[i]);kept++;}
   }
   if(candidates.empty())break;
   auto better=[](auto&a,auto&b){return a.score>b.score;};
   sort(candidates.begin(),candidates.end(),better);
   vector<State<W>>unique;unordered_set<uint64_t>seen;
   for(auto&child:candidates)if(seen.insert(child.hash).second){unique.push_back(child);if((int)unique.size()>=width*max(1,lookfactor))break;}
   if(lookfactor){
    int maxcount=0;for(auto&child:unique)maxcount=max(maxcount,child.count);
    for(auto&child:unique){
     double gain=(variant>=24&&variant<32)?future2(child):future(child);
     const double weights[4]={1.0,.7,1.0,1.5};
     child.score+=gain*weights[variant%4];
     if(variant>=40 && variant<44) {
      const double discounts[4]={.3,.5,2.0,3.0};
      child.score+=gain*(discounts[variant%4]-weights[variant%4]);
     }
     if(variant>=16 && variant<20)child.score+=distance_score(child);
     if(variant>=28 && variant<32)child.score+=configurationscore(child);
     if(variant>=44 && variant<48) {
      // Completing a pattern exposes a fresh target; evaluate future production
      // instead of dropping all latent-state quality at submission boundaries.
      if(child.count==maxcount && child.base==child.count*targetsize) {
       child.score+=gain*.5;
      }
     }
    }
    sort(unique.begin(),unique.end(),better);
    if(variant>=56 && variant<60) {
     // Stochastic beam sampling from near-best candidates avoids repeatedly
     // locking all beam slots onto the same greedy construction order.
     const double temperature[4]={.15,.3,.6,1.0};
     mt19937_64 random(seed+depth*7919);
     for(auto&child:unique){
      double u=(double((random()>>11)+1))*0x1.0p-53;
      child.score-=temperature[variant%4]*log(-log(u));
     }
     sort(unique.begin(),unique.end(),better);
    }
   }
   beam.clear();
   unordered_map<uint64_t,int>diversity;
   for(auto&child:unique){
    if((int)beam.size()>=width)break;
    if(variant>=20 && variant<24) {
     uint64_t key=hash64(child.count);for(int w=0;w<W;w++)key=hash64(key^child.matter.a[w]);
     if(diversity[key]>=4)continue;diversity[key]++;
    }
    if((variant<4 || variant>=8)&&!visited.insert(child.hash).second)continue;
    paths.push_back({child.path,child.op});child.path=paths.size()-1;beam.push_back(child);
   }
  }
  cout<<"FAIL\n";return 1;
 }
};
int main(){int r,count,width,steps,variant,cap,look;uint64_t seed;double seconds;
 if(!(cin>>r>>count>>width>>steps>>variant>>cap>>look>>seed>>seconds))return 2;
 vector<string>pattern(2*r+1);for(auto&s:pattern)cin>>s;
 switch((1+3*(r-1)*r+63)/64){
 #define CASE(N) case N:{Planner<N>planner(r,count,width,steps,variant,cap,look,seed,seconds,pattern);return planner.solve();}
 CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7)
 }return 2;
}
