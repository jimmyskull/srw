#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <map>
#include <unordered_map>
#include <fstream>
#include <string>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>

#include <set>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <igraph/igraph.h>

#include <cstdlib>

typedef unsigned long int user_id;

typedef void * Users_data;

struct endpoint {
  endpoint(user_id v2, unsigned data) : v2(v2), data(data) {}

  ~endpoint() {}

  user_id v2;
  unsigned data;
};

class Graph {
public:
  typedef std::unordered_map<user_id, std::vector<endpoint>> nei_t;
  typedef std::vector<endpoint>::iterator iterator;

  Graph(unsigned edges, unsigned vertex_count, bool allocation) : vertex_count(vertex_count) {
    if (allocation) {
      adj.reserve(vertex_count);
    }
  }

  Graph() : vertex_count(0) {}
  ~Graph() {}

  void ReadFile(std::string filename) {
    std::ifstream infile(filename);
    std::string temp;
    std::set<user_id> ids;

    std::cout << "Reading edge list " << filename << std::endl;
    if (infile.is_open()) {
      while (std::getline(infile, temp)) {
        std::istringstream buffer(temp);
        std::vector<user_id> line((std::istream_iterator<user_id>(buffer)),
                                  std::istream_iterator<user_id>());

        if (line.size() != 2) {
          std::cerr << "Expected two numbers, got '" << temp << "'" << std::endl;
          exit(1);
        }
        user_id from = line.at(0);
        user_id to = line.at(1);
        adj[from].push_back({to, std::max(rand() % 12 + 1, 1)});
        adj[to].push_back({from, std::max(rand() % 12 + 1, 1)});
        // std::cout << from << "->" << to << " (" << adj.at(from).size() << ")" << std::endl;
        ids.insert(from);
        ids.insert(to);
      }
      infile.close();
    } else {
      std::cerr << "Unable to open file!" << std::endl;
    }
    vertex_count = ids.size();
  }

  iterator iterate_outgoing_edges(user_id vertex) {
    return adj[vertex].begin();
  }

  iterator outgoing_end(user_id vertex) {
    return adj[vertex].end();
  }

  unsigned vertex_count;
  nei_t adj;
};


#include "commons.h"
#include "subgraph.h"
#include "params.h"

#include "alglib/optimization.h"

using namespace std;
using namespace alglib;

class PowerMethod
{
    private:
        unsigned nvert, pnum;
        double *pagerank, *last;
        double **deriv, **dlast;

    public:
        static constexpr double alpha = 0.20; // random walk restart parameter
        static constexpr unsigned maxiter = 1000;
        static constexpr double tolerance = 1e-12;
        Subgraph *sub;
        Params *params;

        PowerMethod(Subgraph *s, Params* p, const unsigned nparams)
        {
            sub = s;
            params = p;
            nvert = sub->subgraph->vertex_count;
            pnum = nparams;

            pagerank = new double[nvert]; // pagerank array
            last = new double[nvert];     // pagerank from the last iteration
            // initial uniform distribution of pagerank
            double scale = 1.0 / nvert;
            for (unsigned i = 0; i < nvert; i++)
            {
                pagerank[i] = scale;
                last[i] = 0.0;
            }

            deriv = new double*[nvert];   // derivatives p_u wrt w_k
            dlast = new double*[nvert];   // derivatives p_u wrt w_k
            for (unsigned i = 0; i < nvert; i++)
            {
                deriv[i] = new double[pnum];
                dlast[i] = new double[pnum];
            }
            for (unsigned i = 0; i < nvert; i++)
                for (unsigned k = 0; k < pnum; k++)
                    deriv[i][k] = dlast[i][k] = 0.0;
        }
        ~PowerMethod()
        {
            delete[] last;
            delete[] pagerank;
            for (unsigned i = 0; i < nvert; i++)
            {
                delete[] deriv[i];
                delete[] dlast[i];
            }
            delete[] deriv;
            delete[] dlast;
        }
    // Personalized pagerank starting from vertex start (at index 0)
    void pers_pagerank()
    {
        Graph *graph = sub->subgraph;
        unsigned iter = 0;
        double err = 1.0;
        // We are done when maxiteration is reached
        // or the error is small enough.
        while (iter++ < maxiter && err > tolerance)
        {
            // copy last iteration to last array
            // and clear pagerank array
            #pragma omp parallel for
            for (unsigned i = 0; i < nvert; i++)
            {
                last[i] = pagerank[i];
                pagerank[i] = 0.0;
            }

            // sum up the nodes without outgoing edges ("dangling nodes").
            // their pagerank sum will be uniformly distributed among all nodes.
            double zsum = 0.0;
            #pragma omp parallel for reduction(+:zsum)
            for (unsigned i = 0; i < sub->zerodeg.size(); i++)
                zsum += last[ sub->zerodeg[i] ];
            double nolinks = (1.0-alpha) * zsum / nvert;

            pagerank[0] += alpha; // add teleport probability to the start vertex
            #pragma omp parallel for
            for (unsigned id = 0; id < nvert; id++)
            {
                double update = (1.0-alpha) * last[id];
                Graph::iterator end = graph->outgoing_end(id);
                for (Graph::iterator e = graph->iterate_outgoing_edges(id); e != end; e++)
                {
                    #pragma omp atomic
                    pagerank[(*e).v2] += (update * sub->score(id, (*e).v2));
                }
                #pragma omp atomic
                pagerank[id] += nolinks; // pagerank from "dangling nodes"
            }

            // sum the pagerank
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (unsigned i = 0; i < nvert; i++)
                sum += pagerank[i];

            // normalize to valid probabilities, from 0 to 1.
            sum = 1.0 / sum;
            #pragma omp parallel for
            for (unsigned i = 0; i < nvert; i++)
                pagerank[i] *= sum;

            // sum up the error
            err = 0.0;
            #pragma omp parallel for reduction(+:err)
            for (unsigned i = 0; i < nvert; i++)
                err += fabs(pagerank[i] - last[i]);

            //cout << "Iteration " << iter << endl;
            //cout << "Error: " << err << endl;
        }
        //cout << "PageRank iterations: " << iter << endl;
    }

    void derivatives()
    {
        Graph *graph = sub->subgraph;
        for (unsigned k = 0; k < pnum; k++)
        {
            unsigned iter = 0;
            bool stop = false;
            // We are done when maxiteration is reached
            // or the error is small enough.
            while (iter < maxiter && !stop)
            {
                iter++;

                // copy last iteration
                #pragma omp parallel for
                for (unsigned i = 0; i < nvert; i++)
                {
                    dlast[i][k] = deriv[i][k];
                    deriv[i][k] = 0.0;
                }

                // std::cout << "********"<< nvert << std::endl;
                #pragma omp parallel for
                for (unsigned id = 0; id < nvert; id++)
                {
                    Graph::iterator end = graph->outgoing_end(id);
                    // std::cout << ">>>>>>>> "<< end - graph->iterate_outgoing_edges(id) << std::endl;
                    for (Graph::iterator e = graph->iterate_outgoing_edges(id); e != end; e++)
                    {
                        user_id fr = (*e).v2;
                        double calc = sub->score(id, fr) * dlast[id][k]
                                      +
                                      pagerank[id] * (1.0-alpha) * params->qderiv(id, fr, k);
                        #pragma omp atomic
                        deriv[fr][k] += calc;
                    }
                }

                stop = true;
                for (unsigned d = 0; d < sub->positive.size(); d++)
                {
                    unsigned pos = sub->positive[d];

                    if (fabs(deriv[pos][k] - dlast[pos][k]) > tolerance)
                        stop = false;
                }
                for (unsigned l = 0; l < sub->negative.size(); l++)
                {
                    unsigned neg = sub->negative[l];

                    if (fabs(deriv[neg][k] - dlast[neg][k]) > tolerance)
                        stop = false;
                }
            }
            //cout << "Derivative iterations: " << iter << endl;
        }
    }
    inline double get_pagerank(unsigned u) { return pagerank[u]; }
    inline double get_derivative(unsigned u, unsigned w) { return deriv[u][w]; }
};

class Optimizer
{
    private:
        static constexpr double B = 1.0;

        vector<Subgraph*> subgraphs;
        vector<Params*> parameters;
        vector<PowerMethod*> powers;

        inline double hloss(double x) { if (x < 0) return 0; else return logistic(x / B);
        }
        inline double dhloss(double x) { return logistic_deriv(x / B) / B; }

        void candidate_sums(unsigned s, double& sum_p, vector<double>& sum_d)
        {
            Subgraph *sg = subgraphs[s];
            PowerMethod *power = powers[s];

            for (unsigned d = 0; d < sg->positive.size(); d++)
            {
                unsigned pos = sg->positive[d];
                sum_p += power->get_pagerank(pos);
                for (unsigned w = 0; w < sum_d.size(); w++)
                    sum_d[w] += power->get_derivative(pos, w);
            }
            for (unsigned l = 0; l < sg->negative.size(); l++)
            {
                unsigned neg = sg->negative[l];
                sum_p += power->get_pagerank(neg);
                for (unsigned w = 0; w < sum_d.size(); w++)
                    sum_d[w] += power->get_derivative(neg, w);
            }
        }

    public:
        static const unsigned PNUM = 5;
        vector<double> wvec;

    double Fw(vector<double> &drv)
    {
        double norm = dot_product(wvec, wvec);

        for (unsigned w = 0; w < drv.size(); w++)
            drv[w] += 2*wvec[w];

        double sum = 0.0;
        for (unsigned s = 0; s < subgraphs.size(); s++)
        {
            Subgraph *sg = subgraphs[s];
            Params *params = parameters[s];
            PowerMethod *power = powers[s];

            params->set_wvec(wvec);

            sg->recompute_scores(*params);
            params->recalculate_derivs();

            power->pers_pagerank();
            power->derivatives();

            double sum_p = 0.0;
            vector<double> sum_d(wvec.size(), 0.0);
            candidate_sums(s, sum_p, sum_d);

            double loss = 0.0;
            for (unsigned d = 0; d < sg->positive.size(); d++)
            {
                unsigned pos = sg->positive[d];
                double p_d = power->get_pagerank(pos);
                for (unsigned l = 0; l < sg->negative.size(); l++)
                {
                    unsigned neg = sg->negative[l];
                    double p_l = power->get_pagerank(neg);

                    double diff = p_l/sum_p - p_d/sum_p;
                    loss += hloss(diff);
                    for (unsigned w = 0; w < wvec.size(); w++)
                        drv[w] += dhloss(diff) * (
                                    (power->get_derivative(neg, w)*sum_p - p_l*sum_d[w])/(sum_p*sum_p)
                                    -
                                    (power->get_derivative(pos, w)*sum_p - p_d*sum_d[w])/(sum_p*sum_p)
                                  );
                }
            }
            std::cout <<"LOSS="<<loss<<"  SUM="<<sum<<std::endl;
            sum += loss;
        }
        return norm + sum;
    }

    Optimizer(Graph *graph, Users_data *profile, vector<user_id>& users)
    {
        wvec = vector<double>(PNUM, 0.0);
        for (unsigned u = 0; u < users.size(); u++)
        {
            user_id id = users[u];
            Subgraph* sub = new Subgraph(graph, id, TIMEPOINT);
            Params* prm   = new Params(wvec, sub->subgraph, sub->mutual);
            PowerMethod* power = new PowerMethod(sub, prm, PNUM);

            subgraphs.push_back(sub);
            parameters.push_back(prm);
            powers.push_back(power);
        }
    }
    ~Optimizer()
    {
        for (unsigned i = 0; i < subgraphs.size(); i++)
            delete subgraphs[i];
        for (unsigned i = 0; i < parameters.size(); i++)
            delete parameters[i];
        for (unsigned i = 0; i < powers.size(); i++)
            delete powers[i];
    }

    // starts optimization
    void run()
    {
        double w[PNUM] = {0, 0, 0, 0, 0};
        real_1d_array x;
        x.setcontent(PNUM, w);

        double epsg = 10e-8, epsf = 0, epsx = 0;
        ae_int_t maxits = 0;
        minlbfgsstate state;
        minlbfgsreport rep;

        minlbfgscreate(PNUM, x, state);
        minlbfgssetcond(state, epsg, epsf, epsx, maxits);
        minlbfgsoptimize(state, &optimize, 0, (void*)this);
        minlbfgsresults(state, x, rep);

        cout << "Optimization iterations: " << rep.iterationscount << ", termtype = " << rep.terminationtype << endl;
        cout << (x.tostring(8).c_str()) << endl;
    }

    //alglib callback
    static void optimize(const real_1d_array &w, double &func, real_1d_array &grad, void *ptr)
    {
        Optimizer *opt = (Optimizer*)ptr;
        cout << "Called: [";
        //set wvec
        for (unsigned i = 0; i < opt->wvec.size(); i++)
        {
            opt->wvec[i] = w[i];
            cout << w[i] << ", ";
        }
        cout << "]" << endl;

        vector<double> g(opt->wvec.size(), 0.0);
        func = opt->Fw(g);
        cout << "Function value: " << func << endl;

        for (unsigned i = 0; i < opt->wvec.size(); i++)
        {
            grad[i] = g[i];
            cout << "G[" << i << "] = " << grad[i] << ", ";
        }
        cout << endl;
    }
};

class Predictor
{
    private:
        vector<double> wvec;
        vector<double>  auc;
        vector<unsigned> total, added;

        void evaluate(PowerMethod *power)
        {
            Subgraph* sub = power->sub;
            unordered_map<user_id, bool> future;
            vector<pair<double, user_id> > scores;

            // no friends added
            if (sub->positive.size() == 0)
                return;

            for (unsigned i = 0; i < sub->positive.size(); i++)
            {
                user_id id = sub->positive[i];
                future[id] = true;
                scores.push_back( make_pair(power->get_pagerank(id), id) );
            }
            for (unsigned i = 0; i < sub->negative.size(); i++)
            {
                user_id id = sub->negative[i];
                scores.push_back( make_pair(power->get_pagerank(id), id) );
            }
            sort(scores.begin(), scores.end());

            unsigned total_added = sub->added.size();

            vector<PredictionAnswer> pav;
            for (unsigned i = 0; i < scores.size(); i++)
            {
                PredictionAnswer pa;
                pa.prediction = scores[i].first;
                pa.answer = future.count(scores[i].second) > 0;
                pav.push_back(pa);
            }
            auc.push_back(calculateAUC(pav));

            total.push_back(total_added);
            added.push_back(sub->positive.size());
        }

        void report(const char *output)
        {
            ofstream off(output);
            off << "total\tfuture\tAUC" << endl;
            for (unsigned i = 0; i < auc.size(); i++)
                off << total[i] << "\t" << added[i] << "\t" << auc[i] << endl;
            off.close();
        }

    public:
        Predictor(vector<double> w)
        {
            wvec = w;
        }
        void run(Graph *graph, Users_data *profile, vector<user_id>& users, const char* outfile)
        {
            for (unsigned u = 0; u < users.size(); u++)
            {
                user_id id = users[u];
                Subgraph* sub = new Subgraph(graph, id, TIMEPOINT);
                Params* prm   = new Params(wvec, sub->subgraph, sub->mutual);
                sub->recompute_scores(*prm);

                PowerMethod* power = new PowerMethod(sub, prm, wvec.size());
                power->pers_pagerank();
                evaluate(power);

                // output after every 5000 users
                if (u % 5000 == 0)
                    report(outfile);

                delete power;
                delete prm;
                delete sub;
            }
            report(outfile);
        }
};


Graph *load_graph_data() {
  Graph *g = new Graph();
  //g->ReadFile("pa.el");
  g->ReadFile("banana.el");
  return g;
}

int main(int argc, char *argv[])
{
    srand(time(0));

    Graph *graph = load_graph_data();

    vector<user_id> users, training;
    // The list with selected users IDs.
    load_ids("banana-ids.txt", users);
    //load_ids("ids.txt", users);

    // Use first 100 for training, rest for testing
    for (unsigned i = 0; i < 80; i++)
        training.push_back(users[i]);
    Optimizer opt(graph, 0, training);
    opt.run();

    // Optimizer::PNUM is the number of edge features.
    // Features are computed in params.h:edge_features.

    // This is an vector already computed by the Optimizer
    // so we just use it for guiding the random walks.
    //
    //double x[Optimizer::PNUM] = {0.983772, 2.6467, 1.35106, 1.21225, -3.74018};
    //vector<double> wvec(x, x+Optimizer::PNUM);

    // vector<user_id> testing(users.begin() + 100, users.end());
    // Predictor pred(opt.wvec);
    // pred.run(graph, 0, testing, "predictions.txt");

    delete graph;
    return 0;
}

