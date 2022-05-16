
mkdir build && cd build
cmake ..

# Pour compiler seulement le benchmark

make bench

# Pour compiler la solution

make harris

# Sinon

make all

# Pour lancer le benchmark

./bench

# Pour tester une version
# (une option help est disponible)

./harris -m GPU -o filename

# Et idem pour les autres versions.
# Privilégier les images de petites/moyennes tailles étant donné que nous n'avons pas pu tester les grosses images.
 
# Bibliothèques

https://github.com/gabime/spdlog
https://github.com/CLIUtils/CLI11
https://github.com/LuaDist/libpng
https://github.com/google/benchmark

# Contributeurs
Julien CROS
Théau DEGROOTE
Jordan FAILLOUX
Nicolas ROMANO